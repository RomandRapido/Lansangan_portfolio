package com.example.csv_data_processor

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.boot.autoconfigure.domain.EntityScan
import org.springframework.data.jpa.repository.config.EnableJpaRepositories
import org.springframework.data.elasticsearch.repository.config.EnableElasticsearchRepositories

@SpringBootApplication
@EnableJpaRepositories(basePackages = ["com.example.csv_data_processor.repository"])
@EnableElasticsearchRepositories(basePackages = ["com.example.csv_data_processor.repository.search"])
@EntityScan(basePackages = ["com.example.csv_data_processor.model.entity"])
class CsvDataProcessorApplication

fun main(args: Array<String>) {
	runApplication<CsvDataProcessorApplication>(*args)
}