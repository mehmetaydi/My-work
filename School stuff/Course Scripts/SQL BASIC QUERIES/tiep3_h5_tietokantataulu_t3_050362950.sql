select distinct artist.artist_id, first_name ,last_name FROM artist ,artwork
where artwork.artist_id = artist.artist_id and
(technique = 'drawing' or 'painting' ) 
ORDER BY artist.artist_id ASC;

-- Tietokantojen perusteet 2020
-- H5 T3
-- mehmet.aydin@tuni.fi 