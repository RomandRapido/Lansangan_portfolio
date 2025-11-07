
package com.example.csv_data_processor.service

import com.example.csv_data_processor.model.document.SalesDocument
import com.example.csv_data_processor.repository.search.SalesSearchRepository
import org.springframework.data.elasticsearch.core.ElasticsearchOperations
import org.springframework.data.elasticsearch.core.SearchHits
import org.springframework.data.elasticsearch.core.query.Criteria
import org.springframework.data.elasticsearch.core.query.CriteriaQuery
import org.springframework.stereotype.Service
import java.time.LocalDateTime
import org.springframework.data.elasticsearch.core.aggregation.AggregatedPage
import org.springframework.data.elasticsearch.core.query.NativeSearchQueryBuilder
import org.elasticsearch.index.query.QueryBuilders
import org.elasticsearch.search.aggregations.AggregationBuilders
import org.elasticsearch.search.aggregations.bucket.terms.Terms
import org.elasticsearch.search.aggregations.metrics.Avg

@Service
class SearchService(
    private val salesSearchRepository: SalesSearchRepository,
    private val elasticsearchOperations: ElasticsearchOperations
) {

    fun searchByCustomerName(name: String): List<SalesDocument> {
        return salesSearchRepository.findByCustomerNameContaining(name)
    }

    fun searchByCategory(category: String): List<SalesDocument> {
        return salesSearchRepository.findByCategory(category)
    }

    fun searchBySalesGreaterThan(amount: Double): List<SalesDocument> {
        return salesSearchRepository.findBySalesGreaterThan(amount)
    }

    fun searchByRegion(region: String): List<SalesDocument> {
        val criteria = Criteria.where("region").`is`(region)
        val query = CriteriaQuery(criteria)
        val searchHits: SearchHits<SalesDocument> = elasticsearchOperations.search(query, SalesDocument::class.java)
        return searchHits.map { it.content }.toList()
    }

    fun searchByDateRange(startDate: LocalDateTime, endDate: LocalDateTime): List<SalesDocument> {
        val criteria = Criteria.where("orderDate").between(startDate, endDate)
        val query = CriteriaQuery(criteria)
        val searchHits: SearchHits<SalesDocument> = elasticsearchOperations.search(query, SalesDocument::class.java)
        return searchHits.map { it.content }.toList()
    }

    fun searchByProductName(name: String): List<SalesDocument> {
        val criteria = Criteria.where("productName").contains(name)
        val query = CriteriaQuery(criteria)
        val searchHits: SearchHits<SalesDocument> = elasticsearchOperations.search(query, SalesDocument::class.java)
        return searchHits.map { it.content }.toList()
    }

    fun advancedSearch(
        customerName: String?,
        category: String?,
        region: String?,
        minSales: Double?,
        maxSales: Double?
    ): List<SalesDocument> {
        var criteria = Criteria()

        customerName?.let {
            criteria = criteria.and(Criteria.where("customerName").contains(it))
        }
        category?.let {
            criteria = criteria.and(Criteria.where("category").`is`(it))
        }
        region?.let {
            criteria = criteria.and(Criteria.where("region").`is`(it))
        }
        if (minSales != null && maxSales != null) {
            criteria = criteria.and(Criteria.where("sales").between(minSales, maxSales))
        } else if (minSales != null) {
            criteria = criteria.and(Criteria.where("sales").greaterThanEqual(minSales))
        } else if (maxSales != null) {
            criteria = criteria.and(Criteria.where("sales").lessThanEqual(maxSales))
        }

        val query = CriteriaQuery(criteria)
        val searchHits: SearchHits<SalesDocument> = elasticsearchOperations.search(query, SalesDocument::class.java)
        return searchHits.map { it.content }.toList()
    }

    fun getTotalSalesByCategory(): Map<String, Double> {
        val aggregationName = "sales_by_category"

        val searchQuery = NativeSearchQueryBuilder()
            .withQuery(QueryBuilders.matchAllQuery())
            .addAggregation(
                AggregationBuilders
                    .terms(aggregationName)
                    .field("category")
                    .subAggregation(AggregationBuilders.sum("total_sales").field("sales"))
            )
            .build()

        val searchHits = elasticsearchOperations.search(searchQuery, SalesDocument::class.java)
        val aggregations = searchHits.aggregations

        val categoryAgg = aggregations?.get(aggregationName) as? Terms
        val results = mutableMapOf<String, Double>()

        categoryAgg?.buckets?.forEach { bucket ->
            val totalSales = bucket.aggregations.get("total_sales") as? org.elasticsearch.search.aggregations.metrics.Sum
            results[bucket.keyAsString] = totalSales?.value ?: 0.0
        }

        return results
    }

    fun getAverageProfitByRegion(): Map<String, Double> {
        val aggregationName = "profit_by_region"

        val searchQuery = NativeSearchQueryBuilder()
            .withQuery(QueryBuilders.matchAllQuery())
            .addAggregation(
                AggregationBuilders
                    .terms(aggregationName)
                    .field("region")
                    .subAggregation(AggregationBuilders.avg("avg_profit").field("profit"))
            )
            .build()

        val searchHits = elasticsearchOperations.search(searchQuery, SalesDocument::class.java)
        val aggregations = searchHits.aggregations

        val regionAgg = aggregations?.get(aggregationName) as? Terms
        val results = mutableMapOf<String, Double>()

        regionAgg?.buckets?.forEach { bucket ->
            val avgProfit = bucket.aggregations.get("avg_profit") as? Avg
            results[bucket.keyAsString] = avgProfit?.value ?: 0.0
        }

        return results
    }
}