
// drop keyspace restaurant_kp;

// create key space for restaurant reviews application
create keyspace if not exists restaurant_kp with replication={
  'class': 'SimpleStrategy', 'replication_factor': 3} ;

use restaurant_kp;

// create table for query one
create table restaurant_reviews(
  customer_name text,
  restaurant_name text,
  review text,
  review_date date,
  primary key ((restaurant_name), review_date, customer_name)
) WITH CLUSTERING ORDER BY (review_date desc, customer_name asc);

// create table for query two
create table restaurant_zipcode(
  restaurant_name text,
  zipcode text,
  rating float,
  primary key ((zipcode), rating, restaurant_name)
) WITH CLUSTERING ORDER BY (rating desc, restaurant_name asc);

// create table for query three
create table restaurant_cuisine(
  restaurant_name text,
  zipcode text,
  cuisine text,
  primary key ((zipcode, cuisine), restaurant_name)
);


// populate data for restaurant_reviews
insert into restaurant_reviews (customer_name, restaurant_name, review, review_date)
  values (
    'Alice', 'Sweet Donuts Coffee', 'Greate place to have breakfirst or brunch',
    '2015-10-1'
);
insert into restaurant_reviews (customer_name, restaurant_name, review, review_date)
  values (
    'Bob', 'Sweet Donuts Coffee', 'I love the donuts. They are sweet.',
    '2016-9-2'
);
insert into restaurant_reviews (customer_name, restaurant_name, review, review_date)
  values (
    'Trudy', 'Sweet Donuts Coffee', 'Not a big fan of donuts, but the eggs are good.',
    '2016-8-11'
);
insert into restaurant_reviews (customer_name, restaurant_name, review, review_date)
  values (
    'Alice', 'Farmers', 'I LOVe the dinner here.',
    '2015-10-2'
);
insert into restaurant_reviews (customer_name, restaurant_name, review, review_date)
  values (
    'Alice', 'Joes Roasted Ribs', 'I cannot be more satisfied ! Ribs!',
    '2015-10-4'
);
insert into restaurant_reviews (customer_name, restaurant_name, review, review_date)
  values (
    'Alice', 'Smart Chicken Express', 'Err, dump name but good food.',
    '2015-10-5'
);


// populate data for restaurant_zipcode
insert into restaurant_zipcode (
  restaurant_name, zipcode, rating
)
values(
  'Sweet Donuts Coffee', 'TN 38111', 4.6
);
insert into restaurant_zipcode (
  restaurant_name, zipcode, rating
)
values(
  'Farmers', 'TN 38111', 4.9
);
insert into restaurant_zipcode (
  restaurant_name, zipcode, rating
)
values(
  'Joes Roasted Ribs', 'TN 38111', 4.0
);
insert into restaurant_zipcode (
  restaurant_name, zipcode, rating
)
values(
  'Smart Chicken Express', 'AR 72034', 3.9
);
// populate data for restaurant_cuisine
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Sweet Donuts Coffee', 'TN 38111', 'Roasted Eggs Without Yogs');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Sweet Donuts Coffee', 'TN 38111', 'Roasted Yogs');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Sweet Donuts Coffee', 'TN 38111', 'French Roasts');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Sweet Donuts Coffee', 'TN 38111', 'Just a normal Sandwich');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Sweet Donuts Coffee', 'TN 38111', 'Just a pancake with fruits');

insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Farmers', 'TN 38111', 'Steak Diane');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Farmers', 'TN 38111', 'Turkey In Wine Sauce');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Farmers', 'TN 38111', 'French Roasts');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Farmers', 'TN 38111', 'Just a normal Sandwich');
insert into restaurant_cuisine (restaurant_name, zipcode, cuisine)
  values ('Farmers', 'TN 38111', 'Just a pancake with fruits');

select customer_name, review, review_date from restaurant_reviews
where restaurant_name = 'Sweet Donuts Coffee'
order by review_date desc;

select restaurant_name from restaurant_zipcode
where zipcode = 'TN 38111'
order by rating desc;

select restaurant_name from restaurant_cuisine
where zipcode = 'TN 38111' and cuisine = 'French Roasts';
