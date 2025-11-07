package com.example.csv_data_processor.model.entity

import jakarta.persistence.*

@Entity
@Table(name = "products")
data class Product(
    @Id
    @Column(name = "product_id")
    val productId: String,

    @Column(name = "product_name", nullable = false)
    val productName: String,

    @Column(name = "category", length = 50)
    val category: String,

    @Column(name = "sub_category", length = 50)
    val subCategory: String,

    @OneToMany(mappedBy = "product")
    val orderItems: MutableList<OrderItem> = mutableListOf()
)
