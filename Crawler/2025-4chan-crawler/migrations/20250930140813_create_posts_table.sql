-- Add migration script here
CREATE TABLE posts(
    -- id BIGSERIAL NOT NULL,
    board_name TEXT NOT NULL,
    thread_number BIGINT NOT NULL,
    post_number BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    data JSONB NOT NULL
);

CREATE UNIQUE INDEX ON posts (board_name, post_number, created_at);


SELECT create_hypertable('posts', 'created_at', chunk_time_interval => INTERVAL '1 hours');
