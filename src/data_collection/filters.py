def filter_data_analyst_vacancies(vacancies):
    
    filtered = []
    
    for vac in vacancies:
        name = vac.get("name", "").lower()
        
        #if "professional_roles" in vac:
        #   roles = vac["professional_roles"]
        #   role_ids = [str(role.get("id")) for role in roles]
            
        #   analyst_role_ids = ["10", "134", "148", "150", "156", "163", "164"] # аналитические роли
        #   has_analyst_role = any(role_id in analyst_role_ids for role_id in role_ids)

        include_phrases = [
            "аналитик данных",
            "data analyst", 
            "дата аналитик",
            "bi аналитик",
            "bi-аналитик",
            "bi analyst",
            "бизнес-аналитик",
            "системный аналитик",
            "product analyst",
            "web analyst",
            "маркетинг-аналитик",
            "финансовый аналитик",
            "продуктовый аналитик"
        ]
        
        exclude_words = [
            "юрист", "менеджер", "водитель", "оператор",
            "продавец", "кассир", "бухгалтер", "экономист",
            "консультант", "администратор", "hr", "рекрутер"
        ] # иначе собирает ненужные вакансии
        
        has_include_phrase = any(phrase in name for phrase in include_phrases)
        has_exclude_word = any(word in name for word in exclude_words)
        

        if has_include_phrase and not has_exclude_word:
            filtered.append(vac) 
    
    return filtered


def filter_by_salary(vacancies, min_salary=None, max_salary=None, currency="RUR"):

    if min_salary is None and max_salary is None:
        return vacancies
    
    filtered = []
    
    for vac in vacancies:
        salary = vac.get('salary')
        if not salary:
            continue

        if salary.get('currency') != currency:
            continue
            
        salary_from = salary.get('from')
        salary_to = salary.get('to')
        
        if salary_from is None and salary_to is None:
            continue
            
        effective_salary = salary_from if salary_from is not None else salary_to
        
        if min_salary is not None:
            if effective_salary is not None and effective_salary < min_salary:
                continue
                
        if max_salary is not None:
            if effective_salary is not None and effective_salary > max_salary:
                continue
                
        filtered.append(vac)
    
    return filtered

def filter_by_experience(vacancies, experience_levels=None):
    if not experience_levels:
        return vacancies
    
    filtered = []
    
    for vac in vacancies:
        exp = vac.get('experience', {}).get('id')
        if exp in experience_levels:
            filtered.append(vac)
    
    return filtered

def filter_by_employment_type(vacancies, employment_types=None):

    if not employment_types:
        return vacancies
    
    filtered = []
    
    for vac in vacancies:
        employment = vac.get('employment', {}).get('id')
        if employment in employment_types:
            filtered.append(vac)
    
    return filtered

def filter_by_schedule(vacancies, schedule_types=None):

    if not schedule_types:
        return vacancies
    
    filtered = []
    
    for vac in vacancies:
        schedule = vac.get('schedule', {}).get('id')
        if schedule in schedule_types:
            filtered.append(vac)
    
    return filtered

def remove_duplicates(vacancies):
    seen_ids = set()
    unique_vacancies = []
    
    for vac in vacancies:
        vac_id = vac.get('id')
        if vac_id and vac_id not in seen_ids:
            seen_ids.add(vac_id)
            unique_vacancies.append(vac)
    
    return unique_vacancies

def filter_by_skills(vacancies, required_skills=None, excluded_skills=None):
    if not required_skills and not excluded_skills:
        return vacancies
    
    filtered = []
    
    for vac in vacancies:
        snippet = vac.get('snippet', {})
        requirement = snippet.get('requirement', '').lower()
        
        if required_skills:
            has_all_skills = all(
                skill.lower() in requirement 
                for skill in required_skills
            )
            if not has_all_skills:
                continue

        if excluded_skills:
            has_excluded = any(
                skill.lower() in requirement 
                for skill in excluded_skills
            )
            if has_excluded:
                continue
        
        filtered.append(vac)
    
    return filtered