import streamlit as st
import json
import logging
import urllib3
import time
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import threading

# Configure Streamlit page
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.filter-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)

# Set up logging
@st.cache_resource
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

# Configuration
http = urllib3.PoolManager()
BASE_URL = "https://api.airtable.com/v0/appmhDEU2efjHqHpm"

# Pricing constants for Claude
CLAUDE_COST_PER_INPUT_TOKEN = 0.000003
CLAUDE_COST_PER_OUTPUT_TOKEN = 0.000015

# API Keys
AIRTABLE_API_KEY = os.environ.get('AIRTABLE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

class StreamlitAgenticWorkflow:
    def __init__(self):
        self.anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.token_counter = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0,
            'processed_count': 0,
            'lock': threading.Lock()
        }
    
    def get_airtable_headers(self):
        return {
            'Authorization': f'Bearer {AIRTABLE_API_KEY}',
            'Content-Type': 'application/json'
        }
    
    def verify_airtable_setup(self):
        """Verify Airtable connection and show available views"""
        try:
            url = f"{BASE_URL}/tbldrNIKXchrExJwn"
            response = http.request(
                'GET', url,
                headers=self.get_airtable_headers()
            )
            
            if response.status == 200:
                st.success("✅ Airtable connection successful")
                return True
            else:
                st.error(f"❌ Airtable connection failed: {response.status}")
                return False
                
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
            return False
    
    @st.cache_data(ttl=3600)
    def fetch_all_candidates(_self):
        """Fetch all candidates from Airtable with caching"""
        url = f"{BASE_URL}/tbldrNIKXchrExJwn/listRecords"
        all_candidates = []
        offset = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Fetching candidates from database..."):
            while True:
                query = {
                    "fields": ["skill", "work_experience", "Resume_url", "email", 
                              "Current CTC numeric", "Notice Period", "education", 
                              "Function(demo)", "total_experience_in_years", "Select your role",
                              "Industry(demo)"],
                    "view": "viwbiLAQY8fwHBRxm"
                }
                
                if offset:
                    query["offset"] = offset
                
                try:
                    response = http.request(
                        'POST', url,
                        body=json.dumps(query).encode('utf-8'),
                        headers=_self.get_airtable_headers()
                    )
                    
                    response_data = json.loads(response.data.decode('utf-8'))
                    if 'error' in response_data:
                        st.error(f"Airtable API Error: {response_data['error']}")
                        break
                    
                    records = response_data.get('records', [])
                    all_candidates.extend(records)
                    
                    status_text.text(f"Fetched {len(all_candidates)} candidates...")
                    
                    offset = response_data.get('offset')
                    if not offset:
                        break
                        
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"Error fetching candidates: {str(e)}")
                    break
        
        progress_bar.progress(100)
        status_text.text(f"✅ Successfully fetched {len(all_candidates)} candidates")
        return all_candidates
    
    def fetch_jobs(self, table_id="tblpXdqzHKlXbf8CL", view_id="viw1b6oaGb7KmuzTM"):
        """Fetch all jobs from a specific Airtable view with proper pagination"""
        url = f"{BASE_URL}/{table_id}/listRecords"
        all_jobs = []
        offset = None
        page_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.info(f"🔍 Fetching jobs from view: {view_id}")
        
        with st.spinner("Loading all jobs from the specified view..."):
            while True:
                try:
                    query = {
                        "view": view_id,
                        "pageSize": 100,
                        "fields": [
                            "Role Description", 
                            "function", 
                            "CTC Budget Numeric", 
                            "Experience needed in years", 
                            "Select the role", 
                            "Position & Client Name",
                            "industry"
                        ]
                    }
                    
                    if offset:
                        query["offset"] = offset
                    
                    response = http.request(
                        'POST', 
                        url,
                        body=json.dumps(query).encode('utf-8'),
                        headers=self.get_airtable_headers(),
                        timeout=30
                    )
                    
                    if response.status == 429:
                        st.warning("⚠️ Rate limit reached. Waiting 30 seconds...")
                        time.sleep(30)
                        continue
                    
                    if response.status != 200:
                        error_data = response.data.decode('utf-8')
                        st.error(f"❌ Airtable API Error {response.status}: {error_data}")
                        break
                    
                    response_data = json.loads(response.data.decode('utf-8'))
                    
                    if 'error' in response_data:
                        st.error(f"❌ API Error: {response_data['error']}")
                        break
                    
                    records = response_data.get('records', [])
                    all_jobs.extend(records)
                    page_count += 1
                    
                    progress = min(0.9, page_count * 0.1)
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Fetched {len(all_jobs)} jobs from {page_count} pages...")
                    
                    offset = response_data.get('offset')
                    if not offset:
                        break
                    
                    time.sleep(0.2)
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON decode error: {str(e)}")
                    break
                    
                except Exception as e:
                    st.error(f"❌ Error fetching jobs: {str(e)}")
                    break
        
        progress_bar.progress(1.0)
        
        if all_jobs:
            status_text.text(f"✅ Successfully fetched {len(all_jobs)} jobs from view {view_id}")
            st.success(f"🎉 Found {len(all_jobs)} jobs in the specified view!")
        else:
            status_text.text(f"❌ No jobs found in view {view_id}")
        
        return all_jobs
    
    def get_unique_roles_functions_industries(self, candidates: List[Dict], jobs: List[Dict]) -> Dict:
        """Extract unique roles, functions, and industries from both candidates and jobs"""
        candidate_roles = set()
        candidate_functions = set()
        candidate_industries = set()
        job_roles = set()
        job_functions = set()
        job_industries = set()
        
        # Extract from candidates
        for candidate in candidates:
            fields = candidate.get('fields', {})
            
            # Handle 'Select your role' field
            roles = fields.get('Select your role', [])
            if isinstance(roles, str):
                candidate_roles.add(roles)
            elif isinstance(roles, list):
                candidate_roles.update(roles)
            
            # Handle 'Function(demo)' field
            functions = fields.get('Function(demo)', [])
            if isinstance(functions, str):
                candidate_functions.add(functions)
            elif isinstance(functions, list):
                candidate_functions.update(functions)
            
            # Handle 'Industry(demo)' field
            industries = fields.get('Industry(demo)', [])
            if isinstance(industries, str):
                candidate_industries.add(industries)
            elif isinstance(industries, list):
                candidate_industries.update(industries)
        
        # Extract from jobs
        for job in jobs:
            fields = job.get('fields', {})
            
            # Handle job role
            job_role = fields.get('Select the role')
            if job_role:
                if isinstance(job_role, str):
                    job_roles.add(job_role)
                elif isinstance(job_role, list):
                    job_roles.update(job_role)
            
            # Handle job function
            job_function = fields.get('function')
            if job_function:
                if isinstance(job_function, str):
                    job_functions.add(job_function)
                elif isinstance(job_function, list):
                    job_functions.update(job_function)
            
            # Handle job industry
            job_industry = fields.get('industry')
            if job_industry:
                if isinstance(job_industry, str):
                    job_industries.add(job_industry)
                elif isinstance(job_industry, list):
                    job_industries.update(job_industry)
        
        return {
            'candidate_roles': sorted(list(candidate_roles)),
            'candidate_functions': sorted(list(candidate_functions)),
            'candidate_industries': sorted(list(candidate_industries)),
            'job_roles': sorted(list(job_roles)),
            'job_functions': sorted(list(job_functions)),
            'job_industries': sorted(list(job_industries)),
            'all_roles': sorted(list(candidate_roles.union(job_roles))),
            'all_functions': sorted(list(candidate_functions.union(job_functions))),
            'all_industries': sorted(list(candidate_industries.union(job_industries)))
        }
    
    def search_keywords_in_resume(self, candidate: Dict, keywords: List[str]) -> Dict:
        """Search for keywords in candidate's resume text and return match details"""
        fields = candidate.get('fields', {})
        
        # Get all text fields to search in
        searchable_text = " ".join([
            fields.get('work_experience', ''),
            fields.get('skill', ''),
            fields.get('education', '')
        ]).lower()
        
        matched_keywords = []
        keyword_counts = {}
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in searchable_text:
                matched_keywords.append(keyword)
                keyword_counts[keyword] = searchable_text.count(keyword_lower)
        
        match_percentage = (len(matched_keywords) / len(keywords) * 100) if keywords else 0
        
        return {
            'matched_keywords': matched_keywords,
            'keyword_counts': keyword_counts,
            'match_percentage': round(match_percentage, 1),
            'total_matches': sum(keyword_counts.values())
        }
    
    def filter_candidates_flexible(self, all_candidates: List[Dict], filter_config: Dict) -> List[Dict]:
        """Flexible candidate filtering with logical order"""
        progress_container = st.container()
        
        with progress_container:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Total Candidates", len(all_candidates))
            
            # Step 1: Quality filter (remove low-quality profiles first)
            quality_filtered = []
            if filter_config.get('enable_quality_filter', True):
                for candidate in all_candidates:
                    fields = candidate.get('fields', {})
                    skill = fields.get('skill', '').strip()
                    work_exp = fields.get('work_experience', '').strip()
                    
                    if (skill and work_exp and 
                        len(skill) >= filter_config.get('min_skill_length', 50) and 
                        len(work_exp) >= filter_config.get('min_experience_length', 100) and
                        len(skill.split()) >= 10 and len(work_exp.split()) >= 20):
                        quality_filtered.append(candidate)
            else:
                quality_filtered = all_candidates
            
            with col2:
                st.metric("After Quality", len(quality_filtered))
            
            # Step 2: Apply matching filters (role, function, industry)
            multi_filtered = []
            
            for candidate in quality_filtered:
                fields = candidate.get('fields', {})
                passes_all_filters = True
                
                # Check role filter (if enabled)
                if filter_config.get('enable_role_filter', False) and filter_config.get('selected_roles', []):
                    candidate_roles = fields.get('Select your role', [])
                    if isinstance(candidate_roles, str):
                        candidate_roles = [candidate_roles]
                    elif not isinstance(candidate_roles, list):
                        candidate_roles = []
                    
                    role_match = any(job_role.lower() in [role.lower() for role in candidate_roles] 
                                   for job_role in filter_config['selected_roles'])
                    
                    if not role_match:
                        passes_all_filters = False
                
                # Check function filter (if enabled)
                if filter_config.get('enable_function_filter', False) and filter_config.get('selected_functions', []):
                    candidate_functions = fields.get('Function(demo)', [])
                    if isinstance(candidate_functions, str):
                        candidate_functions = [candidate_functions]
                    elif not isinstance(candidate_functions, list):
                        candidate_functions = []
                    
                    function_match = any(job_func.lower() in [func.lower() for func in candidate_functions] 
                                       for job_func in filter_config['selected_functions'])
                    
                    if not function_match:
                        passes_all_filters = False
                
                # Check industry filter (if enabled)
                if filter_config.get('enable_industry_filter', False) and filter_config.get('selected_industries', []):
                    candidate_industries = fields.get('Industry(demo)', [])
                    if isinstance(candidate_industries, str):
                        candidate_industries = [candidate_industries]
                    elif not isinstance(candidate_industries, list):
                        candidate_industries = []
                    
                    industry_match = any(job_industry.lower() in [industry.lower() for industry in candidate_industries] 
                                       for job_industry in filter_config['selected_industries'])
                    
                    if not industry_match:
                        passes_all_filters = False
                
                if passes_all_filters:
                    multi_filtered.append(candidate)
            
            with col3:
                st.metric("After Match Filters", len(multi_filtered))
            
            # Step 3: Experience filter
            experience_filtered = []
            for c in multi_filtered:
                cand_exp = c['fields'].get('total_experience_in_years')
                if cand_exp is not None and filter_config['exp_min'] <= cand_exp <= filter_config['exp_max']:
                    experience_filtered.append(c)
            
            with col4:
                st.metric("After Experience", len(experience_filtered))
            
            # Step 4: CTC filter
            ctc_filtered = []
            for c in experience_filtered:
                cand_ctc = c['fields'].get('Current CTC numeric')
                if cand_ctc is not None and filter_config['ctc_min'] <= cand_ctc <= filter_config['ctc_max']:
                    ctc_filtered.append(c)
            
            with col5:
                st.metric("After CTC", len(ctc_filtered))
            
            # Step 5: Keyword filter (final detailed analysis)
            keyword_filtered = ctc_filtered
            if filter_config.get('enable_keyword_search', False) and filter_config.get('keywords', []):
                keyword_filtered = []
                keywords = filter_config['keywords']
                min_keyword_matches = filter_config.get('min_keyword_matches', 1)
                
                for c in ctc_filtered:
                    keyword_result = self.search_keywords_in_resume(c, keywords)
                    if len(keyword_result['matched_keywords']) >= min_keyword_matches:
                        c['keyword_match_data'] = keyword_result
                        keyword_filtered.append(c)
            
            with col6:
                st.metric("Final Count", len(keyword_filtered))
        
        return keyword_filtered
    
    def enhance_job_description(self, abstract_jd: str, role: str, experience_years: int, ctc_budget: float) -> Dict:
        """Transform abstract JD into comprehensive role description"""
        prompt = f"""
You are a Job Description Enhancement Agent. Transform this abstract job description into a comprehensive, detailed role description.

Input Details:
- Role: {role}
- Experience Required: {experience_years} years
- CTC Budget: {ctc_budget} LPA
- Abstract JD: {abstract_jd}

Create a comprehensive role description with:
1. Clear role title and summary
2. Detailed responsibilities (5-8 bullet points)
3. Required technical skills (specific technologies, tools, methodologies)
4. Required soft skills and competencies
5. Educational qualifications
6. Industry experience preferences
7. Key performance indicators or success metrics

Return ONLY a JSON object with this structure:
{{
  "role_title": "Enhanced role title",
  "role_summary": "2-3 sentence summary of the role",
  "key_responsibilities": ["responsibility 1", "responsibility 2", ...],
  "technical_skills": ["skill 1", "skill 2", ...],
  "soft_skills": ["skill 1", "skill 2", ...],
  "education_requirements": "Education details",
  "industry_experience": "Preferred industry background",
  "success_metrics": ["metric 1", "metric 2", ...],
  "enhanced_description": "Complete detailed role description text"
}}
"""
        
        try:
            message = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                enhanced_jd = json.loads(json_match.group())
            else:
                enhanced_jd = json.loads(response_text)
            
            self._update_token_counter(message.usage)
            return enhanced_jd
            
        except Exception as e:
            st.error(f"Error enhancing JD: {str(e)}")
            return {"enhanced_description": abstract_jd, "technical_skills": [], "key_responsibilities": []}
    
    def get_ai_score(self, candidate: Dict, enhanced_jd: Dict, custom_prompt: str = None) -> int:
        """Get AI score for candidate fit based on work experience"""
        try:
            fields = candidate.get('fields', {})
            
            if custom_prompt:
                prompt = f"""
{custom_prompt}

CANDIDATE PROFILE DATA:
Work Experience: {fields.get('work_experience', '')}
Skills: {fields.get('skill', '')}
Education: {fields.get('education', '')}
Total Experience: {fields.get('total_experience_in_years', 0)} years
Current CTC: {fields.get('Current CTC numeric', 0)} LPA

ROLE REQUIREMENTS FOR REFERENCE:
{enhanced_jd.get('enhanced_description', '')}

Key Responsibilities Needed: {enhanced_jd.get('key_responsibilities', [])}
Technical Skills Needed: {enhanced_jd.get('technical_skills', [])}

IMPORTANT: You must respond with ONLY a number between 0-100. Do not include any text, explanations, or additional content. Just the score number.
"""
            else:
                prompt = f"""
Rate this candidate's fit for the role based on their work experience (0-100):

ROLE REQUIREMENTS:
{enhanced_jd.get('enhanced_description', '')}

Key Responsibilities Needed:
{enhanced_jd.get('key_responsibilities', [])}

Technical Skills Needed:
{enhanced_jd.get('technical_skills', [])}

CANDIDATE PROFILE:
Work Experience: {fields.get('work_experience', '')}
Skills: {fields.get('skill', '')}
Education: {fields.get('education', '')}

Focus primarily on how well their WORK EXPERIENCE aligns with the job requirements.
Consider:
1. Relevant job titles and responsibilities
2. Technology/tool experience mentioned
3. Industry experience
4. Project complexity and scope
5. Years of relevant experience

IMPORTANT: Respond with ONLY a number between 0-100. No text, no explanations, just the numerical score.
"""
            
            message = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            
            # Enhanced number extraction
            score = None
            number_match = re.search(r'\b(\d+)\b', response_text)
            if number_match:
                score = int(number_match.group(1))
            
            if score is None:
                complex_patterns = [
                    r'(?:score|rating)[:=\s]*(\d+)',
                    r'(\d+)(?:/100|\%)',
                    r'(\d+)\s*(?:out of|/)\s*100'
                ]
                
                for pattern in complex_patterns:
                    match = re.search(pattern, response_text.lower())
                    if match:
                        score = int(match.group(1))
                        break
            
            if score is None:
                start_number = re.match(r'^(\d+)', response_text.strip())
                if start_number:
                    score = int(start_number.group(1))
            
            if score is None:
                all_numbers = re.findall(r'\d+', response_text)
                if all_numbers:
                    for num_str in all_numbers:
                        num = int(num_str)
                        if 0 <= num <= 100:
                            score = num
                            break
            
            if score is None:
                st.warning(f"Could not extract score from AI response: '{response_text[:100]}'")
                score = 50
            else:
                score = max(0, min(100, score))
            
            self._update_token_counter(message.usage)
            return score
            
        except Exception as e:
            st.error(f"Unexpected error getting AI score: {str(e)}")
            return 50
    
    def score_candidates(self, filtered_candidates: List[Dict], enhanced_jd: Dict, custom_prompt: str = None) -> List[Dict]:
        """Score candidates using AI evaluation and keyword matching"""
        scored_candidates = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, candidate in enumerate(filtered_candidates):
            try:
                fields = candidate.get('fields', {})
                
                ai_score = self.get_ai_score(candidate, enhanced_jd, custom_prompt)
                
                keyword_match_data = candidate.get('keyword_match_data', {
                    'matched_keywords': [],
                    'match_percentage': 0,
                    'total_matches': 0
                })
                
                final_score = ai_score
                
                candidate_result = {
                    "candidate_id": candidate["id"],
                    "resume_url": fields.get('Resume_url', ''),
                    "email": fields.get('email', 'No Email'),
                    "ctc": fields.get('Current CTC numeric', 0),
                    "experience_years": fields.get('total_experience_in_years', 0),
                    "notice_period": fields.get('Notice Period', 'Not specified'),
                    "matched_keywords": keyword_match_data['matched_keywords'],
                    "keyword_match_percentage": keyword_match_data['match_percentage'],
                    "keyword_total_matches": keyword_match_data['total_matches'],
                    "ai_score": ai_score,
                    "final_score": round(final_score, 1),
                    "work_experience": fields.get('work_experience', '')[:200] + "..." if len(fields.get('work_experience', '')) > 200 else fields.get('work_experience', ''),
                    "skills": fields.get('skill', '')
                }
                
                scored_candidates.append(candidate_result)
                
                progress = (idx + 1) / len(filtered_candidates)
                progress_bar.progress(progress)
                status_text.text(f"Scoring candidate {idx + 1}/{len(filtered_candidates)} - Score: {final_score}")
                
            except Exception as e:
                st.error(f"Error processing candidate {candidate.get('id', 'unknown')}: {str(e)}")
        
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        progress_bar.progress(100)
        status_text.text(f"✅ Completed scoring {len(scored_candidates)} candidates")
        
        return scored_candidates
    
    def _update_token_counter(self, usage):
        """Update token counter with API usage"""
        with self.token_counter['lock']:
            self.token_counter['input_tokens'] += usage.input_tokens
            self.token_counter['output_tokens'] += usage.output_tokens
            cost = (usage.input_tokens * CLAUDE_COST_PER_INPUT_TOKEN + 
                   usage.output_tokens * CLAUDE_COST_PER_OUTPUT_TOKEN)
            self.token_counter['total_cost'] += cost
            self.token_counter['processed_count'] += 1

    def display_results(self, scored_candidates: List[Dict], filter_config: Dict):
        """Display the results of AI scoring"""
        if not scored_candidates:
            st.warning("No candidates to display.")
            return

        st.header("📊 AI Scoring Results")
        
        # Top 15 candidates
        top_candidates = scored_candidates[:15]
        
        # Create results dataframe
        results_data = []
        for i, candidate in enumerate(top_candidates):
            results_data.append({
                "Rank": i + 1,
                "Email": candidate['email'],
                "AI Score": candidate['ai_score'],
                "Final Score": candidate['final_score'],
                "CTC": f"{candidate['ctc']} LPA",
                "Experience": f"{candidate['experience_years']} years",
                "Notice Period": candidate['notice_period'],
                "Keywords Match": f"{len(candidate['matched_keywords'])}/{len(filter_config.get('keywords', []))}" if filter_config.get('keywords') else "N/A",
                "Resume URL": candidate['resume_url'] if candidate['resume_url'] else "No Resume"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Scored", len(scored_candidates))
        with col2:
            avg_score = sum(c['final_score'] for c in scored_candidates) / len(scored_candidates)
            st.metric("Average Score", f"{avg_score:.1f}")
        with col3:
            st.metric("Top Score", f"{scored_candidates[0]['final_score']}")
        with col4:
            high_scores = len([c for c in scored_candidates if c['final_score'] >= 80])
            st.metric("High Scores (80+)", high_scores)
        
        # Display results table
        st.subheader("🏆 Top 15 Candidates")
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100),
                "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100),
                "Resume URL": st.column_config.LinkColumn("Resume URL", help="Click to open resume")
            },
            hide_index=True
        )
        
        # Detailed candidate cards
        st.subheader("📋 Detailed Candidate Profiles")
        for i, candidate in enumerate(top_candidates[:5]):  # Show top 5 in detail
            with st.expander(f"#{i+1} - {candidate['email']} (Score: {candidate['final_score']})", expanded=(i==0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Work Experience Preview:**")
                    st.text_area(
                        "Experience",
                        candidate['work_experience'],
                        height=100,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    if candidate['matched_keywords']:
                        st.write("**Matched Keywords:**")
                        st.write(", ".join(candidate['matched_keywords']))
                
                with col2:
                    st.metric("AI Score", candidate['ai_score'])
                    st.metric("CTC", f"{candidate['ctc']} LPA")
                    st.metric("Experience", f"{candidate['experience_years']} years")
                    st.write(f"**Notice Period:** {candidate['notice_period']}")
                    
                    if candidate['resume_url']:
                        st.link_button("📄 View Resume", candidate['resume_url'])


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Resume Matcher</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize workflow
    workflow = StreamlitAgenticWorkflow()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        table_id = st.text_input(
            "Jobs Table ID", 
            value="tblpXdqzHKlXbf8CL",
            help="Enter your Airtable jobs table ID"
        )
        
        view_id = st.text_input(
            "Jobs View ID",
            value="viw1b6oaGb7KmuzTM",
            help="Enter your specific Airtable view ID"
        )
        
        st.markdown("---")
        
        if st.button("🔍 Test Airtable Connection"):
            workflow.verify_airtable_setup()
        
        if st.button("🔄 Load All Candidates", type="primary"):
            st.session_state.candidates = workflow.fetch_all_candidates()
        
        if st.button("📋 Load Jobs from Specific View"):
            with st.spinner(f"Loading jobs from view {view_id}..."):
                st.session_state.jobs = workflow.fetch_jobs(table_id=table_id, view_id=view_id)
    
    # Main interface
    if 'candidates' not in st.session_state:
        st.info("👆 Please load candidates from the sidebar first")
        return
    
    if 'jobs' not in st.session_state:
        st.info("👆 Please load available jobs from the sidebar")
        return
    
    # Job Selection
    st.header("1️⃣ Select Job")
    
    jobs = st.session_state.jobs
    if not jobs:
        st.warning("No jobs found. Check your table ID and API keys.")
        return
    
    # Create job options
    job_options = []
    for job in jobs:
        fields = job.get('fields', {})
        position_client = fields.get('Position & Client Name', 'Unnamed Position')
        ctc = fields.get('CTC Budget Numeric', 0)
        exp = fields.get('Experience needed in years', 0)
        
        if ctc >= 100000:
            ctc_display = f"{ctc/100000:.1f}"
        else:
            ctc_display = str(ctc)
        
        job_options.append(f"{position_client} | {ctc_display} LPA | {exp}Y Exp")
    
    # Add search functionality
    st.subheader("🔍 Search Jobs")
    search_term = st.text_input(
        "Search jobs by position, client, or description:",
        placeholder="e.g. Python Developer, TechCorp, Marketing Manager",
        help="Search across position names, client names, and job descriptions"
    )

    # Filter jobs based on search
    if search_term:
        filtered_job_data = []
        
        for i, job in enumerate(jobs):
            fields = job.get('fields', {})
            position_client = fields.get('Position & Client Name', '').lower()
            description = fields.get('Role Description', '').lower()
            search_lower = search_term.lower()
            
            if (search_lower in position_client or 
                search_lower in description):
                filtered_job_data.append({
                    'job': job,
                    'option': job_options[i],
                    'original_index': i
                })
        
        if filtered_job_data:
            st.success(f"Found {len(filtered_job_data)} matching jobs")
            
            selected_filtered_idx = st.selectbox(
                "Choose a job to process:",
                range(len(filtered_job_data)),
                format_func=lambda x: filtered_job_data[x]['option']
            )
            
            selected_job = filtered_job_data[selected_filtered_idx]['job']
            
        else:
            st.warning("No jobs match your search. Please try different keywords.")
            selected_job = None
    else:
        # Show all jobs
        selected_job_idx = st.selectbox(
            "Choose a job to process:",
            range(len(jobs)),
            format_func=lambda x: job_options[x]
        )
        selected_job = jobs[selected_job_idx]

    if selected_job:
        job_fields = selected_job.get('fields', {})
        
        # Display selected job details
        with st.expander("📋 Job Details", expanded=True):
            position_client = job_fields.get('Position & Client Name', 'Not specified')
            ctc_numeric = job_fields.get('CTC Budget Numeric', 0)
            
            # Convert CTC for display
            if ctc_numeric >= 100000:
                ctc_display = f"{ctc_numeric/100000:.1f} LPA"
            else:
                ctc_display = f"{ctc_numeric} LPA"
            
            # Main job info in 3 columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Position & Client", position_client)
            
            with col2:
                st.metric("CTC Budget", ctc_display)
            
            with col3:
                st.metric("Experience Needed", f"{job_fields.get('Experience needed in years', 0)} years")
            
            # Role description in full width
            st.markdown("**Role Description:**")
            role_description = job_fields.get('Role Description', 'No description available')
            st.text_area(
                "Description:",
                role_description,
                height=120,
                disabled=True,
                label_visibility="collapsed"
            )
            
            # Additional info
            additional_info_cols = st.columns(4)
            
            with additional_info_cols[0]:
                if job_fields.get('function'):
                    st.write(f"**Function:** {job_fields.get('function')}")
            
            with additional_info_cols[1]:
                select_role = job_fields.get('Select the role')
                if select_role:
                    st.write(f"**Role Category:** {select_role}")
            
            with additional_info_cols[2]:
                if job_fields.get('industry'):
                    st.write(f"**Industry:** {job_fields.get('industry')}")
            
            with additional_info_cols[3]:
                st.write(f"**Job ID:** {selected_job.get('id', 'Unknown')[:8]}...")
        
        st.markdown("---")
        
        # Enhanced Filter Configuration
        st.header("2️⃣ Configure Advanced Filters")
        
        # Get unique roles, functions, and industries
        if 'candidates' in st.session_state and 'jobs' in st.session_state:
            roles_functions_industries = workflow.get_unique_roles_functions_industries(
                st.session_state.candidates, 
                st.session_state.jobs
            )
        
        # Create filter configuration in a nice layout
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        
        # Row 1: Multiple Filter Options
        st.subheader("🎯 Multi-Filter Strategy")
        st.info("**Enable any combination of filters below. Candidates must match ALL enabled filters.**")
        
        # Role Filter
        col1, col2 = st.columns([1, 2])
        with col1:
            enable_role_filter = st.checkbox(
                "Enable Role Filter",
                value=True,
                help="Filter candidates based on their role"
            )
        
        with col2:
            if enable_role_filter:
                job_role = job_fields.get('Select the role', '')
                auto_selected_roles = []
                if job_role:
                    if isinstance(job_role, str):
                        auto_selected_roles = [job_role]
                    elif isinstance(job_role, list):
                        auto_selected_roles = job_role
                
                selected_roles = st.multiselect(
                    "Select roles to match:",
                    options=roles_functions_industries['all_roles'],
                    default=auto_selected_roles,
                    help="Candidates must have one of these roles"
                )
                
                if auto_selected_roles:
                    st.caption(f"✅ Auto-selected from job: {', '.join(auto_selected_roles)}")
            else:
                selected_roles = []
        
        # Function Filter
        col1, col2 = st.columns([1, 2])
        with col1:
            enable_function_filter = st.checkbox(
                "Enable Function Filter",
                value=False,
                help="Filter candidates based on their function"
            )
        
        with col2:
            if enable_function_filter:
                job_function = job_fields.get('function', '')
                auto_selected_functions = []
                if job_function:
                    if isinstance(job_function, str):
                        auto_selected_functions = [job_function]
                    elif isinstance(job_function, list):
                        auto_selected_functions = job_function
                
                selected_functions = st.multiselect(
                    "Select functions to match:",
                    options=roles_functions_industries['all_functions'],
                    default=auto_selected_functions,
                    help="Candidates must have one of these functions"
                )
                
                if auto_selected_functions:
                    st.caption(f"✅ Auto-selected from job: {', '.join(auto_selected_functions)}")
            else:
                selected_functions = []
        
        # Industry Filter
        col1, col2 = st.columns([1, 2])
        with col1:
            enable_industry_filter = st.checkbox(
                "Enable Industry Filter",
                value=False,
                help="Filter candidates based on their industry"
            )
        
        with col2:
            if enable_industry_filter:
                job_industry = job_fields.get('industry', '')
                auto_selected_industries = []
                if job_industry:
                    if isinstance(job_industry, str):
                        auto_selected_industries = [job_industry]
                    elif isinstance(job_industry, list):
                        auto_selected_industries = job_industry
                
                selected_industries = st.multiselect(
                    "Select industries to match:",
                    options=roles_functions_industries['all_industries'],
                    default=auto_selected_industries,
                    help="Candidates must have one of these industries"
                )
                
                if auto_selected_industries:
                    st.caption(f"✅ Auto-selected from job: {', '.join(auto_selected_industries)}")
            else:
                selected_industries = []

        st.markdown("---")
        
        # Row 2: Keyword Search Filter
        st.subheader("🔍 Keyword Search Filter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_keyword_search = st.checkbox(
                "Enable Keyword Search in Resume Text",
                value=False,
                help="Search for specific keywords in candidate resumes (work experience, skills, education)"
            )
        
        with col2:
            if enable_keyword_search:
                min_keyword_matches = st.number_input(
                    "Minimum Keyword Matches Required",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="How many keywords must be found in candidate's resume"
                )
        
        if enable_keyword_search:
            st.markdown("**Enter Keywords to Search For:**")
            keywords_input = st.text_area(
                "Keywords (one per line or comma-separated):",
                placeholder="Python\nMachine Learning\nAWS\nReact\nSQL\nProject Management",
                height=100,
                help="Enter keywords to search for in candidate resumes. You can use one keyword per line or separate with commas."
            )
            
            # Process keywords
            keywords = []
            if keywords_input.strip():
                # Split by lines first, then by commas
                lines = keywords_input.strip().split('\n')
                for line in lines:
                    if ',' in line:
                        keywords.extend([kw.strip() for kw in line.split(',') if kw.strip()])
                    else:
                        if line.strip():
                            keywords.append(line.strip())
                
                # Remove duplicates and empty strings
                keywords = list(set([kw for kw in keywords if kw]))
            
            if keywords:
                st.success(f"**Searching for {len(keywords)} keywords**: {', '.join(keywords[:10])}" + 
                          (f" ... and {len(keywords)-10} more" if len(keywords) > 10 else ""))
        else:
            keywords = []
            min_keyword_matches = 0
        
        st.markdown("---")
        
        # Row 3: CTC and Experience Ranges
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 CTC Range Configuration")
            
            job_ctc = job_fields.get('CTC Budget Numeric', 10)
            
            ctc_mode = st.radio(
                "CTC Range Mode:",
                ["auto", "custom"],
                format_func=lambda x: "Auto (based on job budget)" if x == "auto" else "Custom range",
                key="ctc_mode"
            )
            
            if ctc_mode == "auto":
                suggested_min = max(0, int(job_ctc * 0.8))
                suggested_max = int(job_ctc * 1.2)
                
                st.info(f"Auto-calculated: {suggested_min} - {suggested_max} LPA")
                st.caption(f"Based on job budget: {job_ctc} LPA (±20%)")
                
                ctc_min, ctc_max = suggested_min, suggested_max
                
            else:
                ctc_range = st.slider(
                    "Select Custom CTC Range:",
                    min_value=0,
                    max_value=200,
                    value=(int(job_ctc * 0.8), int(job_ctc * 1.2)),
                    step=1,
                    help="Set your own CTC range"
                )
                ctc_min, ctc_max = ctc_range
            
            st.success(f"**CTC Filter**: {ctc_min} - {ctc_max} LPA")
        
        with col2:
            st.subheader("⏱️ Experience Range Configuration")
            
            job_exp = job_fields.get('Experience needed in years', 5)
            
            exp_mode = st.radio(
                "Experience Range Mode:",
                ["auto", "custom"],
                format_func=lambda x: "Auto (based on job requirement)" if x == "auto" else "Custom range",
                key="exp_mode"
            )
            
            if exp_mode == "auto":
                exp_suggested_min = max(0, int(job_exp * 0.85))
                exp_suggested_max = int(job_exp * 1.15)
                
                st.info(f"Auto-calculated: {exp_suggested_min} - {exp_suggested_max} years")
                st.caption(f"Based on job requirement: {job_exp} years (±15%)")
                
                exp_min, exp_max = exp_suggested_min, exp_suggested_max
                
            else:
                exp_range = st.slider(
                    "Select Custom Experience Range:",
                    min_value=0,
                    max_value=30,
                    value=(max(0, int(job_exp * 0.85)), int(job_exp * 1.15)),
                    step=1,
                    help="Set your own experience range"
                )
                exp_min, exp_max = exp_range
            
            st.success(f"**Experience Filter**: {exp_min} - {exp_max} years")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 4: Quality Filters
        st.markdown("---")
        st.subheader("✨ Content Quality Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_quality_filter = st.checkbox(
                "Enable Quality Filter",
                value=True,
                help="Filter out candidates with insufficient profile content"
            )
        
        with col2:
            if enable_quality_filter:
                min_skill_length = st.number_input(
                    "Min Skills Text Length",
                    min_value=10,
                    max_value=500,
                    value=50,
                    help="Minimum characters in skills section"
                )
        
        with col3:
            if enable_quality_filter:
                min_experience_length = st.number_input(
                    "Min Experience Text Length",
                    min_value=50,
                    max_value=1000,
                    value=100,
                    help="Minimum characters in work experience section"
                )
        
        st.markdown("---")
        
        # Filter Summary and Preview
        st.header("3️⃣ Filter Summary & Preview")
        
        # Create filter configuration object
        filter_config = {
            'enable_role_filter': enable_role_filter,
            'enable_function_filter': enable_function_filter,
            'enable_industry_filter': enable_industry_filter,
            'selected_roles': selected_roles,
            'selected_functions': selected_functions,
            'selected_industries': selected_industries,
            'enable_keyword_search': enable_keyword_search,
            'keywords': keywords,
            'min_keyword_matches': min_keyword_matches,
            'ctc_min': ctc_min,
            'ctc_max': ctc_max,
            'exp_min': exp_min,
            'exp_max': exp_max,
            'enable_quality_filter': enable_quality_filter,
            'min_skill_length': min_skill_length if enable_quality_filter else 0,
            'min_experience_length': min_experience_length if enable_quality_filter else 0
        }
        
        # Display filter summary
        with st.expander("📋 Filter Summary", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Active Filters:**")
                active_filters = []
                
                if enable_role_filter and selected_roles:
                    active_filters.append(f"• **Role Filter**: {', '.join(selected_roles)}")
                
                if enable_function_filter and selected_functions:
                    active_filters.append(f"• **Function Filter**: {', '.join(selected_functions)}")
                
                if enable_industry_filter and selected_industries:
                    active_filters.append(f"• **Industry Filter**: {', '.join(selected_industries)}")
                
                if not active_filters:
                    st.warning("⚠️ No matching filters enabled!")
                else:
                    for filter_desc in active_filters:
                        st.write(filter_desc)
                
                st.write("**Range Filters:**")
                st.write(f"• CTC: {ctc_min} - {ctc_max} LPA")
                st.write(f"• Experience: {exp_min} - {exp_max} years")
            
            with col2:
                st.write("**Optional Filters:**")
                if enable_keyword_search and keywords:
                    st.write(f"• **Keyword Search**: Enabled")
                    st.write(f"  - Keywords: {len(keywords)} terms")
                    st.write(f"  - Min Matches: {min_keyword_matches}")
                    if len(keywords) <= 5:
                        st.write(f"  - Terms: {', '.join(keywords)}")
                    else:
                        st.write(f"  - Terms: {', '.join(keywords[:5])}... (+{len(keywords)-5} more)")
                else:
                    st.write("• Keyword Search: Disabled")
                
                st.write("**Quality Filters:**")
                if enable_quality_filter:
                    st.write(f"• **Quality Filter**: Enabled")
                    st.write(f"  - Min Skills Length: {min_skill_length} chars")
                    st.write(f"  - Min Experience Length: {min_experience_length} chars")
                else:
                    st.write("• Quality Filter: Disabled")
                
                st.write("**Data Status:**")
                st.write(f"• Total Candidates: {len(st.session_state.candidates)}")
                st.write(f"• Total Jobs: {len(st.session_state.jobs)}")
        
        # Filter Preview Button
        if st.button("🔍 Run Filter Preview", type="primary", use_container_width=True):
            # Validate filter configuration
            has_any_matching_filter = (
                (enable_role_filter and selected_roles) or
                (enable_function_filter and selected_functions) or
                (enable_industry_filter and selected_industries)
            )
            
            if not has_any_matching_filter:
                st.error("❌ Please enable and configure at least one matching filter (Role, Function, or Industry)")
                return
            
            if enable_keyword_search and not keywords:
                st.error("❌ Please enter keywords for keyword search or disable keyword search")
                return
            
            with st.spinner("Filtering candidates with your custom configuration..."):
                filtered_candidates = workflow.filter_candidates_flexible(
                    st.session_state.candidates,
                    filter_config
                )
            
            st.session_state.filtered_candidates = filtered_candidates
            st.session_state.filter_config = filter_config
            
            # Display filtering results
            if len(filtered_candidates) == 0:
                st.error("❌ No candidates match your criteria.")
                
                # Provide suggestions
                st.subheader("💡 Suggestions to get more candidates:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Try widening your ranges:**")
                    st.write(f"• Increase CTC range (currently {ctc_min}-{ctc_max})")
                    st.write(f"• Increase experience range (currently {exp_min}-{exp_max})")
                    if enable_keyword_search:
                        st.write(f"• Reduce minimum keyword matches (currently {min_keyword_matches})")
                
                with col2:
                    st.write("**Try different matching:**")
                    st.write("• Try enabling fewer filters")
                    st.write("• Add more roles/functions/industries to match against")
                    st.write("• Disable some filters to be more inclusive")
            
            elif len(filtered_candidates) > 100:
                st.warning(f"⚠️ {len(filtered_candidates)} candidates found.")
                st.info("💡 This is a large number for AI scoring. Consider narrowing your filters for faster processing and lower costs.")
                
                # Show cost implications
                estimated_cost = len(filtered_candidates) * 0.0002
                estimated_time = len(filtered_candidates) * 2.5 / 60
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⚠️ Candidates", len(filtered_candidates))
                with col2:
                    st.metric("💰 Est. Cost", f"${estimated_cost:.3f}")
                with col3:
                    st.metric("⏱️ Est. Time", f"{estimated_time:.1f} min")
            
            else:
                st.success(f"✅ Perfect! {len(filtered_candidates)} candidates found - ideal for AI scoring!")
                
                # Show cost estimation
                estimated_cost = len(filtered_candidates) * 0.0002
                estimated_time = len(filtered_candidates) * 2.5 / 60
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Candidates", len(filtered_candidates))
                with col2:
                    st.metric("💰 Est. Cost", f"${estimated_cost:.3f}")
                with col3:
                    st.metric("⏱️ Est. Time", f"{estimated_time:.1f} min")
            
            # Show sample candidates for verification with pagination
            if len(filtered_candidates) > 0:
                st.subheader("👥 Filtered Candidates")
                
                # Simple pagination controls
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    page_size_options = [5, 10, 20, 50]
                    candidates_per_page = st.radio(
                        "Candidates per page:",
                        page_size_options,
                        index=1,
                        horizontal=True
                    )
                
                with col2:
                    total_pages = max(1, (len(filtered_candidates) - 1) // candidates_per_page + 1)
                    current_page = st.slider(
                        "Page:",
                        min_value=1,
                        max_value=total_pages,
                        value=1
                    )
                
                with col3:
                    st.metric("Total Pages", total_pages)
                
                # Calculate pagination
                start_idx = (current_page - 1) * candidates_per_page
                end_idx = start_idx + candidates_per_page
                page_candidates = filtered_candidates[start_idx:end_idx]
                
                # Display candidates for current page
                candidate_data = []
                
                for i, candidate in enumerate(page_candidates):
                    fields = candidate.get('fields', {})
                    
                    # Get matching field info for all enabled filters
                    matching_info_parts = []
                    
                    try:
                        if 'filter_config' in st.session_state:
                            filter_cfg = st.session_state.filter_config
                            
                            # Show role matches if role filter was enabled
                            if filter_cfg.get('enable_role_filter', False):
                                roles = fields.get('Select your role', [])
                                if isinstance(roles, str):
                                    roles = [roles]
                                if roles:
                                    matching_info_parts.append(f"Role: {', '.join(roles[:2])}" + ("..." if len(roles) > 2 else ""))
                            
                            # Show function matches if function filter was enabled  
                            if filter_cfg.get('enable_function_filter', False):
                                functions = fields.get('Function(demo)', [])
                                if isinstance(functions, str):
                                    functions = [functions]
                                if functions:
                                    matching_info_parts.append(f"Func: {', '.join(functions[:2])}" + ("..." if len(functions) > 2 else ""))
                            
                            # Show industry matches if industry filter was enabled
                            if filter_cfg.get('enable_industry_filter', False):
                                industries = fields.get('Industry(demo)', [])
                                if isinstance(industries, str):
                                    industries = [industries]
                                if industries:
                                    matching_info_parts.append(f"Ind: {', '.join(industries[:2])}" + ("..." if len(industries) > 2 else ""))
                        
                        matching_display = " | ".join(matching_info_parts) if matching_info_parts else "N/A"
                        
                    except:
                        matching_display = "N/A"
                    
                    # Get keyword match info if available
                    keyword_info = ""
                    if 'keyword_match_data' in candidate:
                        kmd = candidate['keyword_match_data']
                        if kmd['matched_keywords']:
                            keyword_info = f"{len(kmd['matched_keywords'])}/{len(filter_config.get('keywords', []))} keywords"
                        else:
                            keyword_info = "No matches"
                    elif filter_config.get('enable_keyword_search', False):
                        keyword_info = "No matches"
                    else:
                        keyword_info = "N/A"
                    
                    resume_url = fields.get('Resume_url', '')
                    resume_display = resume_url if resume_url else "No Resume"
                    
                    candidate_data.append({
                        "#": start_idx + i + 1,
                        "Email": fields.get('email', 'N/A'),
                        "Matching Fields": matching_display,
                        "Keywords": keyword_info,
                        "CTC": f"{fields.get('Current CTC numeric', 0)} LPA",
                        "Experience": f"{fields.get('total_experience_in_years', 0)} years",
                        "Notice Period": fields.get('Notice Period', 'N/A'),
                        "Resume URL": resume_display
                    })
                
                # Display the table
                if candidate_data:
                    candidates_df = pd.DataFrame(candidate_data)
                    st.dataframe(
                        candidates_df, 
                        use_container_width=True,
                        column_config={
                            "#": st.column_config.NumberColumn("#", width="small"),
                            "Resume URL": st.column_config.LinkColumn("Resume URL", help="Click to open resume")
                        },
                        hide_index=True
                    )
                    
                    # Pagination info
                    st.info(f"Showing {start_idx + 1}-{min(end_idx, len(filtered_candidates))} of {len(filtered_candidates)} candidates")
        
        # AI Scoring Section
        if 'filtered_candidates' in st.session_state and len(st.session_state.filtered_candidates) > 0:
            st.markdown("---")
            st.header("4️⃣ AI Scoring & Matching")
            
            filtered_count = len(st.session_state.filtered_candidates)
            
            if filtered_count <= 100:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"Ready to score {filtered_count} candidates")
                    estimated_cost = filtered_count * 0.0002
                    estimated_time = filtered_count * 2.5 / 60
                    st.write(f"💰 Estimated cost: ${estimated_cost:.3f}")
                    st.write(f"⏱️ Estimated time: {estimated_time:.1f} minutes")
                
                with col2:
                    st.info("📊 **AI-Only Scoring Enabled**")
                    st.write("• Using pure AI evaluation")
                    st.write("• Keyword matching results included")
                    st.write("• Will show top 15 candidates")
                    st.write("• Scores based on work experience fit")
                
                # Custom AI Prompt Section
                st.markdown("---")
                st.subheader("🎯 Custom AI Scoring Prompt (Optional)")
                
                with st.expander("✨ Configure Custom AI Scoring", expanded=False):
                    st.markdown("""
                    **Use this to customize how the AI evaluates candidates based on your specific requirements.**
                    
                    The AI will use your custom prompt to score each candidate's work experience. Your prompt should:
                    - Define what makes a perfect match for this role
                    - Specify key skills, technologies, or experiences to prioritize
                    - Include any specific criteria important for your company/role
                    """)
                    
                    use_custom_prompt = st.checkbox(
                        "Enable Custom AI Scoring Prompt",
                        value=False,
                        help="Check this to use a custom prompt instead of the default scoring logic"
                    )
                    
                    custom_prompt = None
                    if use_custom_prompt:
                        # Custom prompt text area
                        custom_prompt = st.text_area(
                            "Custom AI Scoring Prompt:",
                            value="",
                            height=300,
                            help="Write your custom prompt here. The AI will use this to evaluate each candidate.",
                            placeholder="Example: You are evaluating candidates for a [ROLE]. Score 0-100 based on:\n\nPERFECT MATCH:\n- Specific skill/experience 1\n- Specific skill/experience 2\n\nGOOD MATCH:\n- General requirement 1\n- General requirement 2\n\nFocus on [specific aspects important to you]..."
                        )
                        
                        # Preview section
                        if custom_prompt.strip():
                            st.markdown("**📋 Prompt Preview:**")
                            preview_container = st.container()
                            with preview_container:
                                st.text_area(
                                    "Your prompt will look like this (with candidate data filled in):",
                                    value=f"{custom_prompt}\n\nCANDIDATE PROFILE DATA:\nWork Experience: [Candidate's work experience]\nSkills: [Candidate's skills]\nEducation: [Candidate's education]\nTotal Experience: [X] years\nCurrent CTC: [X] LPA\n\n[Role requirements for reference]\n\nProvide only a number between 0-100 representing the candidate's fit score.",
                                    height=150,
                                    disabled=True
                                )
                    else:
                        st.info("📝 Using default AI scoring logic focusing on work experience alignment with job requirements.")
                
                st.markdown("---")
                
                if st.button("🚀 Start AI Scoring & Matching", type="primary"):
                    with st.spinner("🤖 Running AI Scoring Workflow..."):
                        
                        # Step 1: Enhance Job Description
                        st.info("🤖 Agent 1: Enhancing Job Description...")
                        enhanced_jd = workflow.enhance_job_description(
                            job_fields.get('Role Description', ''),
                            job_fields.get('Select the role', ''),
                            job_fields.get('Experience needed in years', 0),
                            job_fields.get('CTC Budget Numeric', 0)
                        )
                        
                        # Step 2: Score Candidates
                        if use_custom_prompt and custom_prompt.strip():
                            st.info("🤖 Agent 2: Scoring candidates using custom prompt...")
                        else:
                            st.info("🤖 Agent 2: Scoring candidates using AI evaluation...")
                        
                        scored_candidates = workflow.score_candidates(
                            st.session_state.filtered_candidates,
                            enhanced_jd,
                            custom_prompt if use_custom_prompt else None
                        )
                        
                        st.session_state.scored_candidates = scored_candidates
                        st.session_state.enhanced_jd = enhanced_jd
                        
                        # Display token usage
                        token_info = workflow.token_counter
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Input Tokens", f"{token_info['input_tokens']:,}")
                        with col2:
                            st.metric("Output Tokens", f"{token_info['output_tokens']:,}")
                        with col3:
                            st.metric("Total Cost", f"${token_info['total_cost']:.4f}")
                        
                        st.success("✅ AI scoring completed successfully!")
            
            else:
                st.warning("⚠️ Too many candidates for AI scoring. Please narrow your filters to 100 or fewer candidates.")
        
        # Results Display Section
        if 'scored_candidates' in st.session_state:
            st.markdown("---")
            workflow.display_results(st.session_state.scored_candidates, st.session_state.filter_config)

    # Debug Section
    if st.checkbox("🔧 Enable Debug Mode", help="Enable this to debug Function(demo) and Industry(demo) fields"):
        st.markdown("---")
        st.header("🔧 Debug: Function(demo) & Industry(demo) Analysis")
        
        if st.button("🔍 Deep Debug: Check All Field Names"):
            if 'candidates' in st.session_state and st.session_state.candidates:
                st.subheader("🔬 Raw Field Names Analysis")
                
                # Check first 3 candidates for field names
                for i, candidate in enumerate(st.session_state.candidates[:3]):
                    fields = candidate.get('fields', {})
                    
                    st.write(f"**Candidate #{i+1} (ID: {candidate.get('id', 'Unknown')}):**")
                    
                    # Show ALL field names exactly as they appear
                    st.write("**Exact field names:**")
                    for field_name in sorted(fields.keys()):
                        value = fields[field_name]
                        if value:  # Only show fields with data
                            if 'function' in field_name.lower() or 'industry' in field_name.lower():
                                st.success(f"🎯 `{field_name}` = `{value}`")
                            else:
                                st.write(f"• `{field_name}` = `{str(value)[:50]}...`" if len(str(value)) > 50 else f"• `{field_name}` = `{value}`")
                    
                    st.markdown("---")
                
                # Specific check for the exact fields we want
                st.subheader("🎯 Specific Field Check")
                
                function_demo_found = 0
                industry_demo_found = 0
                function_demo_values = []
                industry_demo_values = []
                
                for candidate in st.session_state.candidates[:20]:  # Check first 20
                    fields = candidate.get('fields', {})
                    
                    # Check exact field names
                    if 'Function(demo)' in fields and fields['Function(demo)']:
                        function_demo_found += 1
                        func_val = fields['Function(demo)']
                        if isinstance(func_val, list):
                            function_demo_values.extend(func_val)
                        else:
                            function_demo_values.append(str(func_val))
                    
                    if 'Industry(demo)' in fields and fields['Industry(demo)']:
                        industry_demo_found += 1
                        ind_val = fields['Industry(demo)']
                        if isinstance(ind_val, list):
                            industry_demo_values.extend(ind_val)
                        else:
                            industry_demo_values.append(str(ind_val))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Function(demo) Fields Found", f"{function_demo_found}/20")
                    if function_demo_values:
                        unique_functions = list(set(function_demo_values))
                        st.write("**Sample values:**")
                        for func in unique_functions[:5]:
                            st.write(f"• {func}")
                    else:
                        st.error("No Function(demo) values found!")
                
                with col2:
                    st.metric("Industry(demo) Fields Found", f"{industry_demo_found}/20")
                    if industry_demo_values:
                        unique_industries = list(set(industry_demo_values))
                        st.write("**Sample values:**")
                        for ind in unique_industries[:5]:
                            st.write(f"• {ind}")
                    else:
                        st.error("No Industry(demo) values found!")
                
                # Show a warning if fields are missing
                if function_demo_found == 0 and industry_demo_found == 0:
                    st.error("🚨 **ISSUE FOUND**: Neither Function(demo) nor Industry(demo) fields contain data!")
                    st.info("**Possible solutions:**")
                    st.write("1. Check if these field names exist in your Airtable")
                    st.write("2. Verify the field names are spelled exactly as 'Function(demo)' and 'Industry(demo)'")
                    st.write("3. Check if these fields have data in your Airtable view")
                    st.write("4. Verify your view permissions include these fields")
                
            else:
                st.error("Please load candidates first using the sidebar")


if __name__ == "__main__":
    main()