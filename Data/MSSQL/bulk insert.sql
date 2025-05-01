use DE_AN
go
-- INSERT Category
INSERT INTO Category (ProductCategory, ProductPrice)
VALUES
('Electronics', 12),
('Home', 468),  
('Clothing', 196),
('Books', 398);

-- INSERT REVENUE
BULK INSERT DailyRevenue
FROM 'C:\Users\AD\Desktop\Data projects\python\Decision-making-system\Data\Revenue_table.csv'
WITH (
    FIELDTERMINATOR = ',', 
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2           
);
select * from DailyRevenue

--INSERT CUSTOMER
delete from Customer_INFO
BULK INSERT Customer_INFO
FROM 'C:\Users\AD\Desktop\Data projects\python\Decision-making-system\Data\Customer_Infomation_Final.csv'
WITH (
    FIELDTERMINATOR = ',', 
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2           
);
select * from Customer_INFO
--INSERT ORDER_INFO
delete from ORDER_INFO
BULK INSERT ORDER_INFO
FROM 'C:\Users\AD\Desktop\Data projects\python\Decision-making-system\Data\Order_Infomation_Final.csv'
WITH (
    FIELDTERMINATOR = ',', 
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2           
);
select * from ORDER_INFO

-- INSERT CUSTOMER SEGEMENTATION
delete from Customer_Segmentation
BULK INSERT Customer_Segmentation
FROM 'C:\Users\AD\Desktop\Data projects\python\Decision-making-system\Data\Customer_Segmentation_Final.csv'
WITH (
    FIELDTERMINATOR = ',', 
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2           
);
select * from Customer_Segmentation
-- INSERT REVIEW
BULK INSERT Review
FROM 'C:\Users\AD\Desktop\Data projects\python\Decision-making-system\Data\review.csv'
WITH (
    FIELDTERMINATOR = ',', 
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2           
);
select * from Review

-- TEST FOREGIN KEY
SELECT 
    T.customer_id, 
    T.PurchaseDate, 
    T.ProductCategory, 
    T.ProductPrice, 
    T.Quantity, 
    T.PaymentMethod, 
    T.TotalPrice, 
    C.fullname 
FROM 
    ORDER_INFO T
INNER JOIN 
    Customer_INFO C
ON 
    T.customer_id = C.customer_id;
select * from Category