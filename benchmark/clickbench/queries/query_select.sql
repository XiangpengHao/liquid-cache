SELECT "MobilePhoneModel", COUNT(DISTINCT "UserID") AS u FROM hits WHERE "MobilePhoneModel" <> '' GROUP BY "MobilePhoneModel" ORDER BY u DESC LIMIT 10;
SELECT "UserID" FROM hits WHERE "UserID" = 435090932899640449;
SELECT COUNT(*) FROM hits WHERE "URL" LIKE '%google%';
SELECT "SearchPhrase", MIN("URL"), COUNT(*) AS c FROM hits WHERE "URL" LIKE '%google%' AND "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY c DESC LIMIT 10;
SELECT "SearchPhrase", MIN("URL"), MIN("Title"), COUNT(*) AS c, COUNT(DISTINCT "UserID") FROM hits WHERE "Title" LIKE '%Google%' AND "URL" NOT LIKE '%.google.%' AND "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY c DESC LIMIT 10;
SELECT * FROM hits WHERE "URL" LIKE '%google%' ORDER BY to_timestamp_seconds("EventTime") LIMIT 10;
SELECT "WatchID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"), AVG("ResolutionWidth") FROM hits WHERE "SearchPhrase" <> '' GROUP BY "WatchID", "ClientIP" ORDER BY c DESC LIMIT 10;
