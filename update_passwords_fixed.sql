-- Update user passwords with correct bcrypt hashes
UPDATE users SET hashed_password = '\\\/Xsy7FVTiHh.M8Gj.LW0OMfJbiDChz.6tdRdgg4lr7SyRVFG' WHERE username = 'admin';
UPDATE users SET hashed_password = '\\\/rRIW33CRuXMYMcr8hv/tLBz7kbhWzWcZQV9PFGWC' WHERE username = 'john_doe';
UPDATE users SET hashed_password = '\\\.GDHM9K3bjfhtIcmgsD4kfM16kEWhrXe' WHERE username = 'jane_viewer';