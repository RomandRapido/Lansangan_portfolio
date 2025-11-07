package com.example.csv_data_processor.repository.search

import com.example.csv_data_processor.model.document.SalesDocument

import org.springframework.data.elasticsearch.repository.ElasticsearchRepository
import org.springframework.stereotype.Repository

@Repository
interface SalesSearchRepository : ElasticsearchRepository<SalesDocument, String> {
    fun findByCustomerNameContaining(name: String): List<SalesDocument>
    fun findByCategory(category: String): List<SalesDocument>
    fun findBySalesGreaterThan(amount: Double): List<SalesDocument>
}