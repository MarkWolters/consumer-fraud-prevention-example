create keyspace if not exists transactions WITH replication = {'class': 'NetworkTopologyStrategy', 'Analytics': 1};

use transactions; 
 
CREATE TABLE if not exists credit_card_transactions_by_user (
 account_no text,
 transaction_time timestamp,
 amount double,
 flagged boolean,
 merchant_category_code text,
 pos_id text,
 purchase_location text,
 transaction_id text, 
 user_name text,
 user_address text, 
 PRIMARY KEY (account_no, transaction_time)
) WITH CLUSTERING ORDER BY (transaction_time DESC);

