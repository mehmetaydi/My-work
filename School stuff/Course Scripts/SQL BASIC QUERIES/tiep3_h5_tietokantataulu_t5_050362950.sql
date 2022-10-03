select distinct artist.artist_id, first_name ,last_name FROM artist ,artwork
where artwork.artist_id = artist.artist_id and
(technique = 'painting'  ) and (
artwork.artist_id = '2' or 
artwork.artist_id = '4')
ORDER BY artist.artist_id ASC;

-- Tietokantojen perusteet 2020
-- H5 T5
-- mehmet.aydin@tuni.fi 