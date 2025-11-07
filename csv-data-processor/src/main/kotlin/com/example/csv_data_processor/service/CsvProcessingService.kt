package com.example.csv_data_processor.service

import com.example.csv_data_processor.model.dto.SalesCsvDto
import com.example.csv_data_processor.repository.*
import com.example.csv_data_processor.model.entity.*
import com.example.csv_data_processor.model.document.SalesDocument

import com.example.csv_data_processor.repository.search.SalesSearchRepository
import com.example.csv_data_processor.mapper.SalesMapper


import com.opencsv.bean.CsvToBeanBuilder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import java.io.InputStreamReader
import org.slf4j.LoggerFactory
import org.slf4j.Logger

import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.core.BatchPreparedStatementSetter
import java.sql.PreparedStatement
import java.sql.Timestamp // For LocalDateTime handling

@Service
class CsvProcessingService(
    private val customerRepository: CustomerRepository,
    private val productRepository: ProductRepository,
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val salesSearchRepository: SalesSearchRepository,
    private val salesMapper: SalesMapper
) {
    private val logger: Logger = LoggerFactory.getLogger(CsvProcessingService::class.java)

    @Transactional
    fun processCsvFile(file: MultipartFile, batchSize : Int = 1000): ProcessingResult {
        val startTime = System.currentTimeMillis() // Record start time
        val result = ProcessingResult()

        // Stream parse CSV to avoid OutOfMemory for large files
        InputStreamReader(file.inputStream).use { reader ->
            val csvIterator = CsvToBeanBuilder<SalesCsvDto>(reader)
                .withType(SalesCsvDto::class.java)
                .withIgnoreLeadingWhiteSpace(true)
                .build()
                .iterator()

            val currentBatch = mutableListOf<SalesCsvDto>()

            while (csvIterator.hasNext()) {
                currentBatch.add(csvIterator.next())
                if (currentBatch.size >= batchSize) {
                    processBatch(currentBatch, result)
                    currentBatch.clear()
                }
            }
            if (currentBatch.isNotEmpty()) { // Process any remaining items
                processBatch(currentBatch, result)
            }
        }

        val endTime = System.currentTimeMillis() // Record end time
        result.processingTimeMillis = endTime - startTime // Calculate and set processing time

        return result
    }

    private fun parseCsv(file: MultipartFile): List<SalesCsvDto> {
        InputStreamReader(file.inputStream).use { reader ->
            return CsvToBeanBuilder<SalesCsvDto>(reader)
                .withType(SalesCsvDto::class.java)
                .withIgnoreLeadingWhiteSpace(true)
                .build()
                .parse()
        }
    }

    private fun processBatch(batch: List<SalesCsvDto>, result: ProcessingResult) {
        val validRows = mutableListOf<SalesCsvDto>()
        val invalidRows = mutableMapOf<String, List<String>>()

        // validate rows
        batch.forEach { csvRow ->
            val errors = csvRow.validate()
            if (errors.isNotEmpty()) {
                invalidRows[csvRow.rowId ?: "unknown"] = errors
            } else {
                validRows.add(csvRow)
            }
        }

        // Add validation errors to result
        invalidRows.forEach { (rowId, errors) -> result.addError(rowId, errors) }

        if (validRows.isEmpty()) {
            return // No valid rows in this batch to process
        }

        try{
            val customerIds = validRows.map { it.customerId }.filterNotNull().toSet()
            val productIds = validRows.map { it.productId }.filterNotNull().toSet()
            val orderIds = validRows.map { it.orderId }.filterNotNull().toSet()

            val existingCustomerIds = customerRepository.findByCustomerIdIn(customerIds).map { it.customerId }.toSet()
            val existingProductIds = productRepository.findByProductIdIn(productIds).map { it.productId }.toSet()
            val existingOrderIds = orderRepository.findByOrderIdIn(orderIds).map { it.orderId }.toSet()

            // These are the IDs we need to save
            val newCustomerIds = customerIds.subtract(existingCustomerIds)
            val newProductIds = productIds.subtract(existingProductIds)
            val newOrderIds = orderIds.subtract(existingOrderIds)

            val customersToSave = mutableSetOf<Customer>()
            val productsToSave = mutableSetOf<Product>()
            val ordersToSave = mutableSetOf<Order>()
            val orderItemsToSave = mutableListOf<OrderItem>()
            val elasticsearchDocuments = mutableListOf<SalesDocument>()

            validRows.forEach { csvRow ->
                val bundle = salesMapper.csvToEntities(csvRow)

                if (bundle.customer.customerId in newCustomerIds) {
                    customersToSave.add(bundle.customer)
                }
                if (bundle.product.productId in newProductIds) {
                    productsToSave.add(bundle.product)
                }
                if (bundle.order.orderId in newOrderIds) {
                    ordersToSave.add(bundle.order)
                }
                orderItemsToSave.add(bundle.orderItem)

                elasticsearchDocuments.add(salesMapper.toElasticsearchDocument(csvRow))
            }
            customerRepository.saveAll(customersToSave)
            productRepository.saveAll(productsToSave)
            orderRepository.saveAll(ordersToSave)
            orderItemRepository.saveAll(orderItemsToSave)

            salesSearchRepository.saveAll(elasticsearchDocuments)

            result.addSuccess(validRows.size)
            result.customerDuplicatesCount += existingCustomerIds.size
            result.orderDuplicatesCount += existingOrderIds.size
            result.productDuplicatesCount += existingProductIds.size

        }catch (e: Exception) {
            // If an exception occurs during the batch save, all valid rows in this batch failed
            logger.error("Error processing batch starting with row ${batch.firstOrNull()?.rowId ?: "unknown"}", e)
            batch.forEach { csvRow ->
                result.addError(
                    csvRow.rowId ?: "unknown",
                    listOf("Batch processing failed: ${e.message ?: "Unknown error"}")
                )
            }
        }
    }
}

data class ProcessingResult(
    var successCount: Int = 0,
    val errors: MutableMap<String, List<String>> = mutableMapOf(),
    var processingTimeMillis: Long = 0, // Added property for processing time
    var orderDuplicatesCount: Int = 0,
    var customerDuplicatesCount: Int = 0,
    var productDuplicatesCount: Int = 0
) {
    fun addSuccess(count: Int = 1) { successCount += count }
    fun addError(rowId: String, errors: List<String>) {
        this.errors[rowId] = errors
    }
}