package com.example.csv_data_processor.model.entity

import jakarta.persistence.*
import java.time.LocalDateTime

@Entity
@Table(name = "orders")
data class Order(
    @Id
    @Column(name = "order_id")
    val orderId: String,

    @Column(name = "order_date", nullable = false)
    val orderDate: LocalDateTime,

    @Column(name = "ship_date")
    val shipDate: LocalDateTime?,

    @Column(name = "ship_mode", length = 50)
    val shipMode: String,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "customer_id")
    val customer: Customer,

    @OneToMany(mappedBy = "order", cascade = [CascadeType.ALL])
    val orderItems: MutableList<OrderItem> = mutableListOf()
)