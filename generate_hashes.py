import bcrypt

# Generate bcrypt hashes for the passwords
passwords = {
    'admin': 'admin123',
    'john_doe': 'password123',
    'jane_viewer': 'viewer123'
}

print("-- SQL to update user passwords with correct bcrypt hashes")
print()

for username, password in passwords.items():
    # Generate bcrypt hash (same way your app does it)
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    hash_str = hashed.decode('utf-8')
    
    print(f"UPDATE users SET hashed_password = '{hash_str}' WHERE username = '{username}';")

print()
print("-- Verify")
print("SELECT username, role, LEFT(hashed_password, 30) as hash_preview FROM users;")
