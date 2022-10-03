SELECT artwork.artwork_id, artwork.artwork_name
FROM artwork
WHERE artwork.artwork_id IN
(SELECT displayed_at.artwork_id
FROM displayed_at);

-- Tietokantojen perusteet 2020
-- H6 T1
-- mehmet.aydin@tuni.fi 