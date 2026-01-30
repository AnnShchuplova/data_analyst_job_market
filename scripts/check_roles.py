import requests

def get_all_professional_roles():
    url = "https://api.hh.ru/professional_roles"
    
    response = requests.get(url)
    data = response.json()
    
    categories = data.get('categories', [])
    
    for category in categories:
        
        roles = category.get('roles', [])
        for role in roles:
            role_id = role.get('id')
            role_name = role.get('name')
            
            if "аналит" in role_name.lower():
                print(f"{role_name} (ID: {role_id})")

if __name__ == "__main__":
    get_all_professional_roles()