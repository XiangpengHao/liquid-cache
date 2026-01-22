SELECT COUNT(*)
  FROM hits h
  JOIN (
    SELECT DISTINCT "UserID"
    FROM hits
    WHERE "UserID" % 1000 = 1
    LIMIT 1000
  ) d
  ON h."UserID" = d."UserID";