
package com.example.csv_data_processor.controller

import com.example.csv_data_processor.model.document.SalesDocument
import com.example.csv_data_processor.service.SearchService
import org.springframework.format.annotation.DateTimeFormat
import org.springframework.web.bind.annotation.*
import java.time.LocalDateTime

@RestController
@RequestMapping("/api/search")
class SearchController(private val searchService: SearchService) {

    @GetMapping("/customer")
    fun searchByCustomerName(@RequestParam name: String): List<SalesDocument> {
        return searchService.searchByCustomerName(name)
    }

    @GetMapping("/category/{category}")
    fun searchByCategory(@PathVariable category: String): List<SalesDocument> {
        return searchService.searchByCategory(category)
    }

    @GetMapping("/sales/above")
    fun searchBySalesAbove(@RequestParam amount: Double): List<SalesDocument> {
        return searchService.searchBySalesGreaterThan(amount)
    }

    @GetMapping("/region/{region}")
    fun searchByRegion(@PathVariable region: String): List<SalesDocument> {
        return searchService.searchByRegion(region)
    }

    @GetMapping("/date-range")
    fun searchByDateRange(
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) startDate: LocalDateTime,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) endDate: LocalDateTime
    ): List<SalesDocument> {
        return searchService.searchByDateRange(startDate, endDate)
    }

    @GetMapping("/product")
    fun searchByProductName(@RequestParam name: String): List<SalesDocument> {
        return searchService.searchByProductName(name)
    }

    @GetMapping("/advanced")
    fun advancedSearch(
        @RequestParam(required = false) customerName: String?,
        @RequestParam(required = false) category: String?,
        @RequestParam(required = false) region: String?,
        @RequestParam(required = false) minSales: Double?,
        @RequestParam(required = false) maxSales: Double?
    ): List<SalesDocument> {
        return searchService.advancedSearch(customerName, category, region, minSales, maxSales)
    }

    @GetMapping("/aggregations/total-sales-by-category")
    fun getTotalSalesByCategory(): Map<String, Double> {
        return searchService.getTotalSalesByCategory()
    }

    @GetMapping("/aggregations/avg-profit-by-region")
    fun getAverageProfitByRegion(): Map<String, Double> {
        return searchService.getAverageProfitByRegion()
    }
}