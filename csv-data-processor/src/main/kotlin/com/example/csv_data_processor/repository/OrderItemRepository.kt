package com.example.csv_data_processor.repository

import com.example.csv_data_processor.model.entity.OrderItem

import org.springframework.data.jpa.repository.Query
import org.springframework.stereotype.Repository
import org.springframework.data.jpa.repository.JpaRepository

@Repository
interface OrderItemRepository : JpaRepository<OrderItem, Long> {
    fun findByOrderOrderId(orderId: String): List<OrderItem>

    @Query("SELECT SUM(oi.sales) FROM OrderItem oi")
    fun getTotalSales(): Double?
}