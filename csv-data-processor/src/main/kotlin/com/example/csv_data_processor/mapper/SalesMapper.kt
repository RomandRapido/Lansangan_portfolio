package com.example.csv_data_processor.mapper

import com.example.csv_data_processor.model.dto.SalesCsvDto
import com.example.csv_data_processor.model.entity.*
import com.example.csv_data_processor.model.document.SalesDocument

import org.springframework.stereotype.Component
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@Component
class SalesMapper {

    private val dateFormatter = DateTimeFormatter.ofPattern("MM/dd/yyyy")

    fun csvToEntities(dto: SalesCsvDto): SalesEntityBundle {
        // Parse dates
        val orderDate = parseDate(dto.orderDate)
        val shipDate = dto.shipDate?.let { parseDate(it) }

        // Create Customer
        val customer = Customer(
            customerId = dto.customerId!!,
            customerName = dto.customerName ?: "Unknown",
            segment = dto.segment ?: "Consumer",
            address = Address(
                country = dto.country ?: "",
                city = dto.city ?: "",
                state = dto.state ?: "",
                postalCode = dto.postalCode,
                region = dto.region ?: ""
            )
        )

        // Create Product
        val product = Product(
            productId = dto.productId!!,
            productName = dto.productName ?: "Unknown Product",
            category = dto.category ?: "Other",
            subCategory = dto.subCategory ?: "Other"
        )

        // Create Order
        val order = Order(
            orderId = dto.orderId!!,
            orderDate = orderDate,
            shipDate = shipDate,
            shipMode = dto.shipMode ?: "Standard",
            customer = customer
        )

        // Create Order Item
        val orderItem = OrderItem(
            rowId = dto.rowId?.toLongOrNull() ?: 0L,
            order = order,
            product = product,
            sales = dto.sales?.toDoubleOrNull() ?: 0.0,
            quantity = dto.quantity?.toIntOrNull() ?: 0,
            discount = dto.discount?.toDoubleOrNull() ?: 0.0,
            profit = dto.profit?.toDoubleOrNull() ?: 0.0
        )

        return SalesEntityBundle(customer, product, order, orderItem)
    }

    fun toElasticsearchDocument(dto: SalesCsvDto): SalesDocument {
        val sales = dto.sales?.toDoubleOrNull() ?: 0.0
        val profit = dto.profit?.toDoubleOrNull() ?: 0.0

        return SalesDocument(
            id = dto.rowId ?: "",
            orderId = dto.orderId ?: "",
            orderDate = parseDate(dto.orderDate),
            shipDate = dto.shipDate?.let { parseDate(it) },
            shipMode = dto.shipMode ?: "",
            customerId = dto.customerId ?: "",
            customerName = dto.customerName ?: "",
            segment = dto.segment ?: "",
            country = dto.country ?: "",
            city = dto.city ?: "",
            state = dto.state ?: "",
            region = dto.region ?: "",
            productId = dto.productId ?: "",
            productName = dto.productName ?: "",
            category = dto.category ?: "",
            subCategory = dto.subCategory ?: "",
            sales = sales,
            quantity = dto.quantity?.toIntOrNull() ?: 0,
            discount = dto.discount?.toDoubleOrNull() ?: 0.0,
            profit = profit,
            profitMargin = if (sales > 0) (profit / sales) * 100 else 0.0
        )
    }

    private fun parseDate(dateString: String?): LocalDateTime {
        return try {
            LocalDateTime.parse(dateString, dateFormatter)
        } catch (e: Exception) {
            LocalDateTime.now()
        }
    }
}

// Bundle to return multiple entities
data class SalesEntityBundle(
    val customer: Customer,
    val product: Product,
    val order: Order,
    val orderItem: OrderItem
)