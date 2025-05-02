-- Data migration for candidate data

DO $$
DECLARE
    alex_id INTEGER;
    emma_id INTEGER;
    daniel_id INTEGER;
    sophia_id INTEGER;
    nathan_id INTEGER;
    olivia_id INTEGER;
    ethan_id INTEGER;
    maya_id INTEGER;
    skill_id INTEGER;
BEGIN
    -- Insert Alex Thompson
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Alex Thompson', 
        'Melbourne, Victoria, Australia',
        'Highly motivated, results oriented Software Engineering Manager with over a decade of experience and proven track record in technology, delivery and people leadership.'
    ) RETURNING id INTO alex_id;

    -- Alex Thompson - Work Experience
    INSERT INTO work_experience (candidate_id, company_name, role, start_date, end_date, is_current, responsibilities)
    VALUES 
    (alex_id, 'PageUp', 'Engineering Manager', '2021-10-01', NULL, TRUE, 'Leading multiple software development teams focused on short to mid-term revenue growth, mentoring and coaching managers, building high-performing teams, defining product and technology roadmaps, stakeholder management, implementing Agile processes, and advocating for platform health.'),
    (alex_id, 'PageUp', 'Engineering Team Lead', '2020-11-01', '2021-09-30', FALSE, 'Planned and delivered $50M revenue-generating SaaS recruitment system, implemented agile practices, built stakeholder relationships, mentored engineers, and aligned team objectives with company vision.'),
    (alex_id, 'carsales.com.au', 'Engineering Team Lead', '2019-04-01', '2020-11-01', FALSE, NULL),
    (alex_id, 'carsales.com.au', 'Senior Software Engineer', '2018-02-01', '2019-04-01', FALSE, NULL),
    (alex_id, 'FLIP Group', 'Senior Software Engineer', '2017-03-01', '2018-02-01', FALSE, NULL),
    (alex_id, 'IFS', 'Senior Software Engineer', '2014-10-01', '2017-02-01', FALSE, NULL),
    (alex_id, 'IFS', 'Software Engineer', '2010-11-01', '2014-10-01', FALSE, NULL);

    -- Alex Thompson - Education
    INSERT INTO education (candidate_id, institution, degree, field_of_study, start_year, end_year)
    VALUES
    (alex_id, 'Curtin University', 'Bachelor of Science', 'Computer Systems and Networking', 2006, 2009),
    (alex_id, 'University of Colombo', 'Bachelor of Science', 'Physics and Statistics', 2007, 2011);

    -- Alex Thompson - Skills
    -- Technical Skills
    INSERT INTO skills (name, category) VALUES ('C#', 'Technical') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'C#' AND category = 'Technical';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Java', 'Technical') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Java' AND category = 'Technical';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Oracle', 'Technical') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Oracle' AND category = 'Technical';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Microservices Architecture', 'Technical') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Microservices Architecture' AND category = 'Technical';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('DevOps', 'Technical') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'DevOps' AND category = 'Technical';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    -- Management Skills
    INSERT INTO skills (name, category) VALUES ('Technology Leadership', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Technology Leadership' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Team Leadership', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Team Leadership' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Mentoring', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Mentoring' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Coaching', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Coaching' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Stakeholder Management', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Stakeholder Management' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Product Roadmapping', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Product Roadmapping' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Process Improvement', 'Management') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Process Improvement' AND category = 'Management';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    -- Methodology Skills
    INSERT INTO skills (name, category) VALUES ('Agile Delivery', 'Methodology') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Agile Delivery' AND category = 'Methodology';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Kanban', 'Methodology') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Kanban' AND category = 'Methodology';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);
    
    INSERT INTO skills (name, category) VALUES ('Agile Implementation', 'Methodology') ON CONFLICT (name, category) DO NOTHING;
    SELECT id INTO skill_id FROM skills WHERE name = 'Agile Implementation' AND category = 'Methodology';
    INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (alex_id, skill_id);

    -- Alex Thompson - Achievements
    INSERT INTO achievements (candidate_id, description)
    VALUES
    (alex_id, 'Led real-time market intelligence tool development for car dealers'),
    (alex_id, 'Re-architected legacy applications to microservices'),
    (alex_id, 'Built cloud-based logistics management platform'),
    (alex_id, 'Designed Field Service Management solutions');

    -- Insert candidate: Emma Collins
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Emma Collins', 
        'Seattle, Washington, USA',
        'Passionate data scientist with expertise in machine learning and AI, focusing on building scalable solutions that turn complex data into actionable insights.'
    ) RETURNING id INTO emma_id;

    -- Insert candidate: Daniel Wilson
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Daniel Wilson', 
        'London, United Kingdom',
        'Experienced product manager with a track record of launching successful software products and leading cross-functional teams to deliver innovative solutions.'
    ) RETURNING id INTO daniel_id;

    -- Insert candidate: Sophia Garcia
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Sophia Garcia', 
        'Toronto, Canada',
        'Full-stack developer specializing in web applications and cloud architecture with a passion for creating intuitive user experiences and robust backend systems.'
    ) RETURNING id INTO sophia_id;

    -- Insert candidate: Nathan Rodriguez
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Nathan Rodriguez',
        'Austin, Texas, USA',
        'Cybersecurity expert with extensive experience in vulnerability assessment, threat detection, and implementing robust security frameworks for enterprise systems.'
    ) RETURNING id INTO nathan_id;

    -- Insert candidate: Olivia Martinez
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Olivia Martinez',
        'Berlin, Germany', 
        'UX/UI designer with a focus on creating accessible, user-centered digital experiences that bridge business goals with user needs through iterative design processes.'
    ) RETURNING id INTO olivia_id;

    -- Insert candidate: Ethan Parker
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Ethan Parker',
        'Singapore',
        'DevOps engineer specializing in CI/CD pipelines, infrastructure as code, and cloud-native solutions that enhance development efficiency and system reliability.'
    ) RETURNING id INTO ethan_id;

    -- Insert candidate: Maya Johnson
    INSERT INTO candidates (name, current_location, summary)
    VALUES (
        'Maya Johnson',
        'Sydney, Australia',
        'Project manager with agile and scrum certification, known for delivering complex technical projects on time and within budget while maintaining team morale and stakeholder satisfaction.'
    ) RETURNING id INTO maya_id;
END $$;