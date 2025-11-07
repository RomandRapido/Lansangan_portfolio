# CSV Data Processor

A high-performance Spring Boot application for processing and analyzing CSV sales data with dual storage: PostgreSQL for relational data and Elasticsearch for advanced search and analytics.

*Note: This is a demo project and not intended for production use. Configuration is not production ready. Instead, this project demonstrates the core concepts of a high-performance Spring Boot application.*

## Overview

This project provides a robust solution for ingesting, validating, and storing large CSV files containing sales transaction data. It features batch processing, data normalization, and real-time search capabilities through a RESTful API with powerful Elasticsearch integration.

## Features

- **Batch CSV Processing**: Processes large CSV files with configurable batch sizes
- **Dual Storage Architecture**:
  - PostgreSQL for normalized relational data (Customers, Orders, Products, Order Items)
  - Elasticsearch for full-text search, filtering, and real-time analytics
- **Data Validation**: Comprehensive validation with detailed error reporting
- **Duplicate Detection**: Automatically handles duplicate customers, products, and orders
- **RESTful API**: Simple endpoints for file upload, data management, and advanced search
- **Performance Metrics**: Tracks processing time and success/failure counts
- **Advanced Search**: Full-text search, filtering, aggregations, and analytics via Elasticsearch

## Architecture

### Data Model

**Relational Entities (PostgreSQL):**
- **Customer**: Customer information with embedded address
- **Product**: Product catalog with categories
- **Order**: Order headers with shipping details
- **OrderItem**: Individual line items with sales metrics

**Search Document (Elasticsearch):**
- **SalesDocument**: Denormalized document with all sales data for fast querying, analytics, and aggregations

### Key Components

- **CsvProcessingService**: Core batch processing engine with validation
- **SearchService**: Elasticsearch query and aggregation service
- **SalesMapper**: Converts CSV rows to entities and search documents
- **Repositories**: JPA repositories for PostgreSQL and Elasticsearch repositories for search

## Getting Started

### Prerequisites

- Java 17 or higher
- Docker and Docker Compose (for infrastructure)
- Gradle 8.x

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd csv-data-processor
   ```

2. **Start infrastructure services**
   ```bash
   docker-compose up -d
   ```

   This starts:
    - PostgreSQL on port 5432
    - Elasticsearch on ports 9200/9300
    - pgAdmin on port 5050 (optional, for database management)

3. **Configure application**

   Edit `application.properties` and update database credentials:
   ```properties
   spring.datasource.url=jdbc:postgresql://localhost:5432/csv_db
   spring.datasource.username=postgres
   spring.datasource.password=password
   ```

4. **Build and run**
   ```bash
   ./gradlew bootRun
   ```

   Or build a JAR:
   ```bash
   ./gradlew build
   java -jar build/libs/csv-data-processor-0.0.1-SNAPSHOT.jar
   ```

5. **Verify Elasticsearch is running**
   ```bash
   curl http://localhost:9200
   ```

## API Endpoints

### CSV Processing

#### Upload CSV File

**POST** `/api/csv/upload`

Uploads and processes a CSV file, storing data in both PostgreSQL and Elasticsearch.

**Parameters:**
- `file` (multipart/form-data): The CSV file to process
- `batchSize` (optional, default: 1000): Number of rows to process per batch

**Example:**
```
bash
curl -X POST http://localhost:8080/api/csv/upload \
  -F "file=@/path/to/sales-data.csv" \
  -F "batchSize=2000"
```
**Response:**
```
json
{
  "successCount": 9994,
  "errors": {},
  "processingTimeMillis": 15234,
  "orderDuplicatesCount": 0,
  "customerDuplicatesCount": 0,
  "productDuplicatesCount": 0
}
```
### Elasticsearch Search Endpoints

#### Search by Customer Name

**GET** `/api/search/customer`

Search for sales records by customer name (partial match supported).

**Parameters:**
- `name` (required): Customer name to search for

**Example:**
```
bash
curl "http://localhost:8080/api/search/customer?name=John"
```
#### Search by Category

**GET** `/api/search/category/{category}`

Find all sales records for a specific product category.

**Example:**
```
bash
curl "http://localhost:8080/api/search/category/Furniture"
```
#### Search by Sales Amount

**GET** `/api/search/sales/above`

Find sales records with sales amount above a threshold.

**Parameters:**
- `amount` (required): Minimum sales amount

**Example:**
```
bash
curl "http://localhost:8080/api/search/sales/above?amount=1000"
```
#### Search by Region

**GET** `/api/search/region/{region}`

Find all sales records for a specific geographic region.

**Example:**
```
bash
curl "http://localhost:8080/api/search/region/West"
```
#### Search by Date Range

**GET** `/api/search/date-range`

Find sales records within a specific date range.

**Parameters:**
- `startDate` (required): Start date in ISO format (yyyy-MM-ddTHH:mm:ss)
- `endDate` (required): End date in ISO format (yyyy-MM-ddTHH:mm:ss)

**Example:**
```
bash
curl "http://localhost:8080/api/search/date-range?startDate=2023-01-01T00:00:00&endDate=2023-12-31T23:59:59"
```
#### Search by Product Name

**GET** `/api/search/product`

Search for sales records by product name (partial match supported).

**Parameters:**
- `name` (required): Product name to search for

**Example:**
```
bash
curl "http://localhost:8080/api/search/product?name=Chair"
```
#### Advanced Search

**GET** `/api/search/advanced`

Perform complex searches with multiple filters.

**Parameters (all optional):**
- `customerName`: Customer name filter
- `category`: Product category filter
- `region`: Geographic region filter
- `minSales`: Minimum sales amount
- `maxSales`: Maximum sales amount

**Example:**
```
bash
curl "http://localhost:8080/api/search/advanced?customerName=John&category=Technology&minSales=500&maxSales=2000&region=East"
```
**Response:**
```
json
[
  {
    "id": "1",
    "orderId": "CA-2016-152156",
    "orderDate": "2023-11-08T10:30:00",
    "customerName": "John Smith",
    "category": "Technology",
    "productName": "HP Laptop",
    "sales": 1499.99,
    "profit": 224.99,
    "profitMargin": 15.0,
    "region": "East"
  }
]
```
### Aggregations & Analytics

#### Total Sales by Category

**GET** `/api/search/aggregations/total-sales-by-category`

Get aggregated total sales amount grouped by product category.

**Example:**
```
bash
curl "http://localhost:8080/api/search/aggregations/total-sales-by-category"
```
**Response:**
```
json
{
  "Furniture": 742000.56,
  "Office Supplies": 719047.03,
  "Technology": 836154.03
}
```
#### Average Profit by Region

**GET** `/api/search/aggregations/avg-profit-by-region`

Get average profit grouped by geographic region.

**Example:**
```
bash
curl "http://localhost:8080/api/search/aggregations/avg-profit-by-region"
```
**Response:**
```
json
{
  "West": 28.67,
  "East": 31.45,
  "Central": 25.89,
  "South": 29.12
}
```
### Debug Endpoints

#### Clear All Data

**DELETE** `/api/debug/clear-all`

Removes all data from PostgreSQL (useful for testing).

*Note: This does NOT clear Elasticsearch data.*

**Example:**
```
bash
curl -X DELETE http://localhost:8080/api/debug/clear-all
```
## CSV Format

The application expects CSV files with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| Row ID | Unique identifier | Yes |
| Order ID | Order identifier | Yes |
| Order Date | Order date (MM/dd/yyyy) | Yes |
| Ship Date | Shipping date (MM/dd/yyyy) | No |
| Ship Mode | Shipping method | No |
| Customer ID | Customer identifier | Yes |
| Customer Name | Customer name | No |
| Segment | Customer segment | No |
| Country/Region | Country | No |
| City | City | No |
| State | State | No |
| Postal Code | Postal code | No |
| Region | Region | No |
| Product ID | Product identifier | Yes |
| Category | Product category | No |
| Sub-Category | Product subcategory | No |
| Product Name | Product name | No |
| Sales | Sales amount | Yes |
| Quantity | Quantity sold | Yes |
| Discount | Discount applied | No |
| Profit | Profit amount | No |

## Configuration

### Application Properties

Key configuration options in `application.properties`:

```properties
# Upload limits
spring.servlet.multipart.max-file-size=10MB
spring.servlet.multipart.max-request-size=10MB

# Elasticsearch
spring.elasticsearch.uris=http://localhost:9200
spring.elasticsearch.connection-timeout=5s
spring.elasticsearch.socket-timeout=60s

# JPA/Hibernate
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
```



### Docker Services

The `docker-compose.yml` file configures:

- **PostgreSQL**: Database with persistent volume
- **Elasticsearch**: Single-node cluster for development (7.17.9)
- **pgAdmin**: Web-based database management UI

Access pgAdmin at `http://localhost:5050` with:
- Email: `admin@admin.com`
- Password: `admin`

Access Elasticsearch at `http://localhost:9200`

### Elasticsearch Index
The application creates an index named sales_data with the following field mappings:
- **Keyword fields**: IDs, category, region, segment (for exact match and aggregations)
- **Text fields**: Customer name, product name, city, state (for full-text search)
- **Numeric fields**: Sales, profit, quantity, discount (for range queries and aggregations)
- **Date fields**: Order date, ship date (for date range queries)
- **Calculated fields**: Profit margin, price range (computed at indexing time)

## Technology Stack

- **Framework**: Spring Boot 3.5.7
- **Language**: Kotlin 1.9.25
- **Database**: PostgreSQL 15
- **Search Engine**: Elasticsearch 7.17.9
- **CSV Processing**: OpenCSV 5.9
- **Build Tool**: Gradle (Kotlin DSL)

## Performance

- Configurable batch processing for memory efficiency
- Bulk insert operations for optimal database performance
- Duplicate detection to avoid unnecessary database operations
- Streaming CSV parsing to handle large files
- Parallel indexing to PostgreSQL and Elasticsearch

**Sample Performance** (10k rows):
- Processing Time: ~30 seconds
- Memory Usage: < 512MB
- Batch Size: 1000 rows
- Elasticsearch indexing: ~5ms per document

## Development

### Project Structure

```
src/main/kotlin/com/example/csv_data_processor/
├── config/          # Application configuration
├── controller/      # REST API endpoints
│   ├── CsvController.kt        # CSV upload endpoints
│   ├── SearchController.kt     # Elasticsearch search endpoints
│   └── DebugController.kt      # Debug utilities
├── model/
│   ├── dto/        # Data transfer objects
│   ├── entity/     # JPA entities
│   └── document/   # Elasticsearch documents
├── mapper/         # DTO to Entity mappers
├── repository/     # Data access layer
│   └── search/     # Elasticsearch repositories
└── service/        # Business logic
├── CsvProcessingService.kt  # CSV processing
└── SearchService.kt         # Elasticsearch queries
```


### Running Tests

```shell script
./gradlew test
```


### Elasticsearch Development Tips

1. **View all indices:**
```shell script
curl http://localhost:9200/_cat/indices?v
```


2. **View index mapping:**
```shell script
curl http://localhost:9200/sales_data/_mapping
```


3. **Count documents:**
```shell script
curl http://localhost:9200/sales_data/_count
```


4. **Search all documents:**
```shell script
curl http://localhost:9200/sales_data/_search?pretty
```


5. **Delete index (careful!):**
```shell script
curl -X DELETE http://localhost:9200/sales_data
```


## Use Cases

This application demonstrates several real-world use cases:

1. **Data Ingestion**: Batch processing of large CSV files from business systems
2. **Full-Text Search**: Finding customers or products by partial name matches
3. **Analytics**: Aggregating sales data by category, region, or time period
4. **Business Intelligence**: Analyzing profit margins and sales performance
5. **Complex Filtering**: Multi-criteria queries combining customer, product, region, and price range
6. **Time-Series Analysis**: Querying sales data within specific date ranges
7. **Performance Reporting**: Real-time aggregations for dashboards and reports

## Known Issues

- Large CSV files (>100MB) may require increasing JVM heap size
- Date parsing expects MM/dd/yyyy format only
- Elasticsearch security is disabled in development configuration
- Debug endpoint for clearing data does not clear Elasticsearch index

---

**Built using Spring Boot and Kotlin**

