SELECT artwork.artwork_id, artwork.artwork_name
FROM artwork
WHERE NOT EXISTS
(SELECT *
FROM displayed_at 
where displayed_at.artwork_id = artwork.artwork_id);

-- Tietokantojen perusteet 2020
-- H6 T2
-- mehmet.aydin@tuni.fi 