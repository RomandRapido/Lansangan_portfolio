package com.example.csv_data_processor.model.dto

import com.opencsv.bean.CsvBindByName
import com.opencsv.bean.CsvDate

data class SalesCsvDto(
    @CsvBindByName(column = "Row ID")
    val rowId: String? = null,

    @CsvBindByName(column = "Order ID")
    val orderId: String? = null,

    @CsvBindByName(column = "Order Date")
    val orderDate: String? = null,

    @CsvBindByName(column = "Ship Date")
    val shipDate: String? = null,

    @CsvBindByName(column = "Ship Mode")
    val shipMode: String? = null,

    @CsvBindByName(column = "Customer ID")
    val customerId: String? = null,

    @CsvBindByName(column = "Customer Name")
    val customerName: String? = null,

    @CsvBindByName(column = "Segment")
    val segment: String? = null,

    @CsvBindByName(column = "Country/Region")
    val country: String? = null,

    @CsvBindByName(column = "City")
    val city: String? = null,

    @CsvBindByName(column = "State")
    val state: String? = null,

    @CsvBindByName(column = "Postal Code")
    val postalCode: String? = null,

    @CsvBindByName(column = "Region")
    val region: String? = null,

    @CsvBindByName(column = "Product ID")
    val productId: String? = null,

    @CsvBindByName(column = "Category")
    val category: String? = null,

    @CsvBindByName(column = "Sub-Category")
    val subCategory: String? = null,

    @CsvBindByName(column = "Product Name")
    val productName: String? = null,

    @CsvBindByName(column = "Sales")
    val sales: String? = null,

    @CsvBindByName(column = "Quantity")
    val quantity: String? = null,

    @CsvBindByName(column = "Discount")
    val discount: String? = null,

    @CsvBindByName(column = "Profit")
    val profit: String? = null
) {
    // Validation method
    fun validate(): List<String> {
        val errors = mutableListOf<String>()

        if (rowId.isNullOrBlank()) errors.add("Row ID is required")
        if (orderId.isNullOrBlank()) errors.add("Order ID is required")
        if (customerId.isNullOrBlank()) errors.add("Customer ID is required")
        if (productId.isNullOrBlank()) errors.add("Product ID is required")

        sales?.toDoubleOrNull() ?: errors.add("Sales must be a number")
        quantity?.toDoubleOrNull() ?: errors.add("Quantity must be a number")

        return errors
    }
}