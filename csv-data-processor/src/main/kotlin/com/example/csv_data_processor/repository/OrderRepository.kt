package com.example.csv_data_processor.repository

import com.example.csv_data_processor.model.entity.Order
import com.example.csv_data_processor.model.entity.Product

import org.springframework.stereotype.Repository
import org.springframework.data.jpa.repository.JpaRepository

@Repository
interface OrderRepository : JpaRepository<Order, String> {
    fun existsByOrderId(orderId: String): Boolean
    fun findByOrderIdIn(orderIds: Set<String>) : List<Order>

}