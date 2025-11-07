package com.example.csv_data_processor.repository

import com.example.csv_data_processor.model.entity.Product

import org.springframework.stereotype.Repository
import org.springframework.data.jpa.repository.JpaRepository

@Repository
interface ProductRepository : JpaRepository<Product, String> {
    fun existsByProductId(productId: String): Boolean
    fun findByCategory(category: String): List<Product>
    fun findByProductIdIn(productIds: Set<String>) : List<Product>

}