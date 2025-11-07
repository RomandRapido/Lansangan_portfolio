package com.example.csv_data_processor.model.entity

import jakarta.persistence.*

@Entity
@Table(name = "order_items")
data class OrderItem(
    @Id
    @Column(name = "row_id")
    val rowId: Long,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "order_id")
    val order: Order,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "product_id")
    val product: Product,

    @Column(name = "sales")
    val sales: Double,

    @Column(name = "quantity")
    val quantity: Int,

    @Column(name = "discount")
    val discount: Double,

    @Column(name = "profit")
    val profit: Double
)