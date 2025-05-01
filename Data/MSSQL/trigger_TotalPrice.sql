-- drop any existing triggers
USE DE_AN
GO
DROP TRIGGER IF EXISTS CalculateTotalPrice;
GO
ALTER TABLE ORDER_INFO
ALTER COLUMN TotalPrice FLOAT NULL;
GO
CREATE TRIGGER CalculateTotalPrice
ON ORDER_INFO
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE o
    SET TotalPrice = i.ProductPrice * i.Quantity
    FROM ORDER_INFO o
    INNER JOIN INSERTED i ON o.OrderID = i.OrderID;
    DECLARE @PurchaseDate DATE;
    DECLARE @TotalRevenue DECIMAL(18, 2);
-- CALCULATE SUM(TotalPrice) FOR EACH DAY THEN INSERT INTO DailyRevenue TABLE
    SELECT @PurchaseDate = PurchaseDate FROM INSERTED;
    SELECT @TotalRevenue = SUM(TotalPrice) 
    FROM ORDER_INFO 
    WHERE PurchaseDate = @PurchaseDate;
    IF EXISTS (SELECT 1 FROM DailyRevenue WHERE RecordedDate = @PurchaseDate)
    BEGIN
        UPDATE DailyRevenue
        SET TotalRevenue = @TotalRevenue
        WHERE RecordedDate = @PurchaseDate;
    END
    ELSE
    BEGIN
        INSERT INTO DailyRevenue (RecordedDate, TotalRevenue)
        VALUES (@PurchaseDate, @TotalRevenue);
    END
END;
GO
--TESTING
INSERT INTO ORDER_INFO (customer_id, PurchaseDate, ProductCategory, ProductPrice, Quantity, PaymentMethod)
VALUES ('KH6969', '2024-01-27', 'Electronics', 100.00, 2, 'Credit Card');

SELECT * FROM ORDER_INFO WHERE customer_id = 'KH6969';
SELECT * FROM DailyRevenue WHERE RecordedDate = '2024-01-27';
SELECT * FROM ORDER_INFO