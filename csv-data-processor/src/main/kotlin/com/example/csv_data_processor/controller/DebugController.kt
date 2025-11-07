package com.example.csv_data_processor.controller

import com.example.csv_data_processor.repository.*

import com.example.csv_data_processor.service.CsvProcessingService
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/debug")
class DebugController(
    private val customerRepo  : CustomerRepository,
    private val orderRepo     : OrderRepository,
    private val orderItemRepo : OrderItemRepository,
    private val productRepo   : ProductRepository
) {
    @DeleteMapping("/clear-all")
    fun clearAll() = run {
        orderItemRepo.deleteAll()
        orderRepo.deleteAll()
        customerRepo.deleteAll()
        productRepo.deleteAll()
        "All data wiped."
    }
}