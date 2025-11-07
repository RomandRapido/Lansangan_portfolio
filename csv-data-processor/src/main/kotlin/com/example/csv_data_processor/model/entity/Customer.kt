package com.example.csv_data_processor.model.entity

import jakarta.persistence.*

@Entity
@Table(name = "customers")
data class Customer(
    @Id
    @Column(name = "customer_id")
    val customerId: String,

    @Column(name = "customer_name", nullable = false)
    val customerName: String,

    @Column(name = "segment", length = 50)
    val segment: String,

    @Embedded
    val address: Address,

    @OneToMany(mappedBy = "customer")
    val orders: MutableList<Order> = mutableListOf()
)

@Embeddable
data class Address(
    @Column(name = "country")
    val country: String,

    @Column(name = "city")
    val city: String,

    @Column(name = "state")
    val state: String,

    @Column(name = "postal_code")
    val postalCode: String?,

    @Column(name = "region")
    val region: String
)
