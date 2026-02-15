
-- Create Database

CREATE DATABASE IF NOT EXISTS recruitment_db
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE recruitment_db;


-- Create Tables 


-- USER TABLE
CREATE TABLE IF NOT EXISTS user (
  id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(150) NOT NULL,
  email VARCHAR(150) NOT NULL,
  password VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL,
  is_approved TINYINT(1) NOT NULL DEFAULT 0,
  PRIMARY KEY (id),
  UNIQUE KEY uq_user_username (username),
  UNIQUE KEY uq_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- CANDIDATE INFO TABLE
CREATE TABLE IF NOT EXISTS candidate_info (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NULL,
  manager_id INT NOT NULL,
  candidate_name VARCHAR(150) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'hired',
  platform VARCHAR(50) NULL,
  role_hired_for VARCHAR(50) NULL,
  reason TEXT NULL,
  score DOUBLE NULL,

  PRIMARY KEY (id),

  KEY idx_candidate_user_id (user_id),
  KEY idx_candidate_manager_id (manager_id),

  CONSTRAINT fk_candidate_user
    FOREIGN KEY (user_id) REFERENCES user (id)
    ON DELETE SET NULL
    ON UPDATE CASCADE,

  CONSTRAINT fk_candidate_manager
    FOREIGN KEY (manager_id) REFERENCES user (id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
