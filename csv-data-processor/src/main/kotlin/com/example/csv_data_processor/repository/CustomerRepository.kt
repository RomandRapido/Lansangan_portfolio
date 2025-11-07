package com.example.csv_data_processor.repository

import com.example.csv_data_processor.model.entity.Customer

import org.springframework.stereotype.Repository
import org.springframework.data.jpa.repository.JpaRepository

@Repository
interface CustomerRepository : JpaRepository<Customer, String> {
    fun existsByCustomerId(customerId: String): Boolean
    fun findBySegment(segment: String): List<Customer>
    fun findByCustomerIdIn(customerIds: Set<String>) : List<Customer>
}