package com.example.csv_data_processor.controller

import com.example.csv_data_processor.service.ProcessingResult

import com.example.csv_data_processor.service.CsvProcessingService
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/csv")
class CsvController(private val csvService: CsvProcessingService) {

    @PostMapping("/upload")
    fun uploadCsv(@RequestParam("file") file: MultipartFile,
                  @RequestParam("batchSize", defaultValue = "1000") batchSize: Int): ProcessingResult {
        return csvService.processCsvFile(file, batchSize)
    }
}