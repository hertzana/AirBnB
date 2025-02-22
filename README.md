# Airbnb

Used XGBoost to predict price of Airbnb listing, using training data with 74K+ data points. Collaborated with Galen Simmons and Nathan Hearnsberger (Deloitte colleagues).

Engineered the following features to increase accuracy of prediction:
1.	Bag of Words NLP to extract key words from listing descriptions
2.	Google Places API to extract data on number of attractions close to the Airbnb listing (i.e. number of restaurants, number of shops, number of hotels, transit stations, etc)
3.	Calculated distance between city center and each Airbnb listing. For example, for NYC, we chose Times Square as the city center and calculated distance between Times Square and each listing in NYC; for Boston, we used Copley as city center; for Washington, D.C., we used the White House as the city center. The idea is that the closer the listing is to the city center, the higher the price
4.	Incorporated median income for each zip code
5.	Get mean R/G/B data from images of rooms
6.	Calculated how long the person has been a host
