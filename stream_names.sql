-- ============================================
-- FULL CLEAN AND RECREATE STREAMS SCRIPT (STATIC)
-- MODEL_CONFIG streams paired by type, fire_detection at the end, cameras last
-- ============================================

BEGIN;

-- 1️⃣ DELETE ALL STREAMS
DELETE FROM public.stream;

-- 2️⃣ DELETE ALL STREAM DEVICES OF TYPE STREAM
DELETE FROM public.device
WHERE type = 'STREAM';

-- 3️⃣ CREATE DEVICES AND STREAMS (STATIC)
INSERT INTO public.device (location, name, type, factory_id)
VALUES
-- Paired MODEL_CONFIG streams (except fire_detection)
('front', 'ppe_detection_cam1sub', 'STREAM', 7),
('back',  'ppe_detection_cam2sub', 'STREAM', 7),

('front', 'surveillance_zones_cam1sub', 'STREAM', 7),
('back',  'surveillance_zones_cam2sub', 'STREAM', 7),

('front', 'time_count_cam1sub', 'STREAM', 7),
('back',  'time_count_cam2sub', 'STREAM', 7),

('front', 'zone_analysis_cam1sub', 'STREAM', 7),
('back',  'zone_analysis_cam2sub', 'STREAM', 7),

('front', 'zone_detection_cam1sub', 'STREAM', 7),
('back',  'zone_detection_cam2sub', 'STREAM', 7),

-- fire_detection paired at the end
('front', 'fire_detection_cam1sub', 'STREAM', 7),
('back',  'fire_detection_cam2sub', 'STREAM', 7),

-- Camera streams themselves (at the very end)
('front', 'cam1sub', 'STREAM', 7),
('back',  'cam2sub', 'STREAM', 7)
RETURNING id, name;

-- 4️⃣ CREATE STREAMS 1:1 FOR THESE DEVICES
INSERT INTO public.stream (iddevice, streamname, streamlink, category)
SELECT id, name, 'http://102.110.12.218:8889', 'STREAM'
FROM public.device
WHERE type = 'STREAM';

COMMIT;

-- 5️⃣ VERIFY RESULT
SELECT d.id AS device_id, d.name AS device_name, s.streamname
FROM device d
LEFT JOIN stream s ON s.iddevice = d.id
WHERE d.type = 'STREAM'
ORDER BY d.id;
