package com.example.csv_data_processor.model.document

import org.springframework.data.annotation.Id
import org.springframework.data.elasticsearch.annotations.*
import java.time.LocalDateTime

@Document(indexName = "sales_data")
data class SalesDocument(
    @Id
    val id: String, // rowId

    @Field(type = FieldType.Keyword)
    val orderId: String,

    @Field(type = FieldType.Date, format = [DateFormat.date_hour_minute_second])
    val orderDate: LocalDateTime,

    @Field(type = FieldType.Date, format = [DateFormat.date_hour_minute_second])
    val shipDate: LocalDateTime?,

    @Field(type = FieldType.Text, analyzer = "standard")
    val shipMode: String,

    // Customer fields
    @Field(type = FieldType.Keyword)
    val customerId: String,

    @Field(type = FieldType.Text, analyzer = "standard", fielddata = true)
    val customerName: String,

    @Field(type = FieldType.Keyword)
    val segment: String,

    // Location fields for geo queries
    @Field(type = FieldType.Text)
    val country: String,

    @Field(type = FieldType.Text)
    val city: String,

    @Field(type = FieldType.Text)
    val state: String,

    @Field(type = FieldType.Keyword)
    val region: String,

    // Product fields
    @Field(type = FieldType.Keyword)
    val productId: String,

    @Field(type = FieldType.Text, analyzer = "standard")
    val productName: String,

    @Field(type = FieldType.Keyword)
    val category: String,

    @Field(type = FieldType.Keyword)
    val subCategory: String,

    // Metrics
    @Field(type = FieldType.Double)
    val sales: Double,

    @Field(type = FieldType.Integer)
    val quantity: Int,

    @Field(type = FieldType.Double)
    val discount: Double,

    @Field(type = FieldType.Double)
    val profit: Double,

    // Calculated fields for better search/analytics
    @Field(type = FieldType.Double)
    val profitMargin: Double? = if (sales > 0) (profit / sales) * 100 else 0.0,

    @Field(type = FieldType.Keyword)
    val priceRange: String? = when {
        sales < 100 -> "Low"
        sales < 500 -> "Medium"
        sales < 1000 -> "High"
        else -> "Premium"
    }
)