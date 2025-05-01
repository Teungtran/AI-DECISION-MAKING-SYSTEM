CREATE DATABASE DE_AN;
GO
USE DE_AN
GO
DROP TABLE IF EXISTS ORDER_INFO;
DROP TABLE IF EXISTS Customer_INFO;
DROP TABLE IF EXISTS DailyRevenue;
DROP TABLE IF EXISTS Review;
DROP TABLE IF EXISTS Category;

CREATE TABLE Category (
    ProductCategory VARCHAR(50) NOT NULL,
    ProductPrice DECIMAL(10, 2),
    CONSTRAINT PK_ProductCategory PRIMARY KEY (ProductCategory)
);
-- Create Customer_INFO table
CREATE TABLE Customer_INFO (
    index_id INT IDENTITY(1,1) PRIMARY KEY NOT NULL,
    customer_id VARCHAR(50) UNIQUE,
    fullname VARCHAR(255) NOT NULL,
    Age INT NOT NULL,          
    Gender VARCHAR(10) NOT NULL, 
    Favoured_Payment_method VARCHAR(50) NOT NULL,
    Favoured_Product_Category VARCHAR(50) NOT NULL
);

-- Create ORDER_INFO table
CREATE TABLE ORDER_INFO (
    OrderID INT IDENTITY(1,1) PRIMARY KEY NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    PurchaseDate DATE NOT NULL,
    ProductCategory VARCHAR(50) NOT NULL,
    ProductPrice FLOAT NOT NULL,
    Quantity INT NOT NULL,
    PaymentMethod VARCHAR(50) NOT NULL,
    TotalPrice FLOAT NOT NULL,
    CONSTRAINT FK_Customer_ID FOREIGN KEY (customer_id) REFERENCES Customer_INFO(customer_id),
    CONSTRAINT FK_ProductCategory FOREIGN KEY (ProductCategory) REFERENCES Category(ProductCategory)
);

-- Create DailyRevenue table
CREATE TABLE DailyRevenue (
    RecordedDate DATE NOT NULL PRIMARY KEY,
    TotalRevenue FLOAT NOT NULL
);

-- Create Review table
CREATE TABLE Review (
    id_feedback INT IDENTITY(1,1) NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    review TEXT NULL,
    CONSTRAINT PK_Review_id_feedback PRIMARY KEY (id_feedback),
    CONSTRAINT FK_Review_customer_id FOREIGN KEY (customer_id) REFERENCES Customer_INFO(customer_id)
);
DROP TABLE IF EXISTS Customer_Segmentation;

CREATE TABLE Customer_Segmentation(
   stt INT IDENTITY(1,1) PRIMARY KEY NOT NULL,
   customer_id VARCHAR(50) NOT NULL,
   TotalSpent FLOAT NOT NULL,
   Frequency INT NOT NULL,
   LastPurchaseDate DATE NOT NULL,
   Recency INT NOT NULL,
   Customer_Labels VARCHAR(50) NOT NULL,
   CONSTRAINT FK_Segments_Customer_ID FOREIGN KEY (customer_id) REFERENCES Customer_INFO(customer_id)
   );
CREATE TABLE conversation_history (
    id INT IDENTITY(1,1) PRIMARY KEY, 
    session_id NVARCHAR(255) NOT NULL,  
    user_question NVARCHAR(MAX) NOT NULL, 
    ai_response NVARCHAR(MAX), 
    created_at DATETIME DEFAULT GETDATE() 
);

-- Query the ORDER_INFO table
SELECT * FROM Customer_Segmentation;
