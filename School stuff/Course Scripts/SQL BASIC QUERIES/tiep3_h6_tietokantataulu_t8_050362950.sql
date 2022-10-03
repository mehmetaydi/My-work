select distinct artwork_name, value ,year_created , first_name, last_name FROM artwork , artist

where (value =
(select max(value) from  artwork ) and artist.artist_id = artwork.artist_id);

-- Tietokantojen perusteet 2020
-- H6 T8
-- mehmet.aydin@tuni.fi 