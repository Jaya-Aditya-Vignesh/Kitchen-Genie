import os
import streamlit as st
import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Kitchen Genie 🧙‍♂️",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to set background image
def add_bg_from_url():
    """
    Add background image from URL or local file
    """
    # Background image URL - using a light cooking-themed image
    bg_image = """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/cooking-ingredients-with-herbs-spices-vegetables_140725-2516.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Add overlay for better text readability */
    .stApp:before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.85);  /* White overlay with 85% opacity */
        z-index: -1;
    }
    
    /* Make content containers slightly transparent to let background show through */
    div.block-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    
    /* Sidebar styling with a different background */
    section[data-testid="stSidebar"] {
        background-color: rgba(240, 240, 245, 0.8);
        border-right: 1px solid #e0e0e0;
    }
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Call the background function
add_bg_from_url()

# Custom CSS for improved UI
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 2.5rem;
    }
    .recipe-card {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.3s;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recipe-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 800;
        color: #2c3e50;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
    }
    .st-emotion-cache-16txtl3 h2 {
        font-weight: 700;
        color: #3498db;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
    }
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(rgba(58, 123, 213, 0.8), rgba(0, 210, 255, 0.8));
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .pantry-title {
        background-color: rgba(248, 249, 250, 0.8);
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .expiry-warning {
        color: #e74c3c;
        font-weight: bold;
    }
    .ingredient-tag {
        background-color: rgba(227, 242, 253, 0.9);
        border-radius: 15px;
        padding: 5px 10px;
        margin: 2px;
        display: inline-block;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(240, 242, 246, 0.8);
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.9);
        font-weight: bold;
    }
    .header-banner {
        background-color: rgba(52, 152, 219, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API')

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if 'selected_recipe' not in st.session_state:
    st.session_state.selected_recipe = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Find Recipes"
if 'recipes_data' not in st.session_state:
    st.session_state.recipes_data = None
if 'favorite_recipes' not in st.session_state:
    st.session_state.favorite_recipes = []
if 'show_expiry_alert' not in st.session_state:
    st.session_state.show_expiry_alert = True
if 'filter_by_diet' not in st.session_state:
    st.session_state.filter_by_diet = "All"
if 'cooking_mode' not in st.session_state:
    st.session_state.cooking_mode = False

# Define callbacks for page navigation
def select_recipe(recipe_metadata):
    st.session_state.selected_recipe = recipe_metadata
    st.session_state.current_tab = "Recipe Details"

def go_to_find_recipes():
    st.session_state.current_tab = "Find Recipes"

def toggle_favorite(recipe_name):
    if recipe_name in st.session_state.favorite_recipes:
        st.session_state.favorite_recipes.remove(recipe_name)
    else:
        st.session_state.favorite_recipes.append(recipe_name)

def toggle_cooking_mode():
    st.session_state.cooking_mode = not st.session_state.cooking_mode

def search_recipes():
    with st.spinner("🔍 Searching for recipes..."):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("dssssproject")

        # Get ingredients nearing expiry
        df = pd.read_excel("grocery_data.xlsx")
        df['days_diff'] = (df['expiry_date'] - df['current_date']).dt.days
        filtered_df = df[(df['days_diff'] >= 0) & (df['days_diff'] <= 7)]

        # Generate query embedding
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_str = "Ingredients: " + ", ".join(filtered_df["Name"])
        query_vector = model.encode([query_str])[0]

        # Query Pinecone and store in session state
        response = index.query(vector=query_vector.tolist(), top_k=8, include_metadata=True)
        st.session_state.recipes_data = response

def add_to_pantry(name, expiry_date, category, quantity, unit):
    """Store ingredient in Excel file with additional metadata"""
    df = pd.DataFrame({
        "Name": [name],
        "expiry_date": [expiry_date],
        "current_date": [datetime.date.today()],
        "category": [category],
        "quantity": [quantity],
        "unit": [unit]
    })

    if not os.path.exists("grocery_data.xlsx"):
        df.to_excel("grocery_data.xlsx", index=False)
    else:
        existing_df = pd.read_excel("grocery_data.xlsx")
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel("grocery_data.xlsx", index=False)

def get_expiry_chart():
    """Generate expiry chart for pantry visualization"""
    if not os.path.exists("grocery_data.xlsx"):
        return None
    
    df = pd.read_excel("grocery_data.xlsx")
    if df.empty:
        return None
        
    df['days_diff'] = (df['expiry_date'] - df['current_date']).dt.days
    
    # Create expiry categories
    conditions = [
        (df['days_diff'] < 0),
        (df['days_diff'] >= 0) & (df['days_diff'] <= 2),
        (df['days_diff'] > 2) & (df['days_diff'] <= 7),
        (df['days_diff'] > 7)
    ]
    categories = ['Expired', 'Use immediately', 'Use soon', 'Fresh']
    df['expiry_status'] = pd.cut(df['days_diff'], bins=[-float('inf'), 0, 2, 7, float('inf')], labels=categories)
    
    # Create the chart with a more attractive style
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-pastel')
    ax = sns.countplot(x='expiry_status', data=df, palette='RdYlGn')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), 
                   textcoords = 'offset points')
    
    plt.title('Pantry Ingredients by Expiry Status', fontsize=14, fontweight='bold')
    plt.xlabel('Expiry Status', fontsize=12)
    plt.ylabel('Number of Items', fontsize=12)
    plt.tight_layout()
    
    # Save the chart to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

def get_category_distribution():
    """Generate category distribution chart for pantry visualization"""
    if not os.path.exists("grocery_data.xlsx"):
        return None
    
    df = pd.read_excel("grocery_data.xlsx")
    if df.empty or 'category' not in df.columns:
        return None
        
    # Create the chart with improved styling
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-pastel')
    category_counts = df['category'].value_counts()
    
    # Use a nicer colormap
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(category_counts)))
    ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), 
                   textcoords = 'offset points')
    
    plt.title('Pantry Items by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Items', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the chart to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

# Fix missing import
import numpy as np

def main():
    # Sidebar for pantry management - always visible
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/cooking-book.png", width=100)
        st.header("🧙‍♂️ Kitchen Genie")
        
        tabs = st.tabs(["🗄️ Pantry", "📊 Analytics", "⚙️ Settings"])
        
        with tabs[0]:
            st.markdown("<div class='pantry-title'>Add Ingredients</div>", unsafe_allow_html=True)
            
            name = st.text_input("Ingredient Name")
            col1, col2 = st.columns(2)
            with col1:
                quantity = st.number_input("Quantity", min_value=0.1, step=0.1, value=1.0)
            with col2:
                unit = st.selectbox("Unit", ["kg", "g", "liters", "ml", "pieces", "cups", "tbsp", "tsp"])
            
            category = st.selectbox("Category", 
                ["Vegetables", "Fruits", "Dairy", "Meat", "Seafood", "Grains", "Spices", "Condiments", "Other"])
            expiry_date = st.date_input("Expiry Date", datetime.date.today() + datetime.timedelta(days=7))
            
            if st.button("➕ Add to Pantry", use_container_width=True):
                add_to_pantry(name, expiry_date, category, quantity, unit)
                st.success(f"Added {name} to pantry!")
            
            st.divider()
            
            # Show pantry contents
            if os.path.exists("grocery_data.xlsx"):
                with st.expander("🥕 View Pantry Contents", expanded=True):
                    df = pd.read_excel("grocery_data.xlsx")
                    df['days_diff'] = (df['expiry_date'] - df['current_date']).dt.days
                    
                    # Show expiring soon items
                    expiring_soon = df[(df['days_diff'] >= 0) & (df['days_diff'] <= 3)]
                    if not expiring_soon.empty and st.session_state.show_expiry_alert:
                        st.warning("⚠️ Items expiring soon:")
                        for _, row in expiring_soon.iterrows():
                            st.markdown(f"**{row['Name']}**: {row['days_diff']} days left")
                        if st.button("Dismiss Alert"):
                            st.session_state.show_expiry_alert = False
                    
                    # Allow filtering and display the table
                    category_filter = st.multiselect("Filter by category", 
                                                      df['category'].unique().tolist() if 'category' in df.columns else [], 
                                                      default=[])
                    if category_filter:
                        filtered_df = df[df['category'].isin(category_filter)]
                    else:
                        filtered_df = df
                    
                    # Display table with styling
                    if not filtered_df.empty:
                        # Add expiry indicator
                        filtered_df['Status'] = filtered_df['days_diff'].apply(
                            lambda x: '🔴 Expired' if x < 0 else 
                                    ('🟠 Use now' if x <= 2 else 
                                    ('🟡 Use soon' if x <= 7 else '🟢 Fresh')))
                        
                        display_cols = ['Name', 'quantity', 'unit', 'category', 'expiry_date', 'Status']
                        display_cols = [col for col in display_cols if col in filtered_df.columns]
                        st.dataframe(filtered_df[display_cols], hide_index=True, use_container_width=True)
                    else:
                        st.info("No items match the selected filters.")
            else:
                st.info("Your pantry is empty. Add ingredients to get started!")
        
        with tabs[1]:
            st.markdown("<div class='pantry-title'>Pantry Analytics</div>", unsafe_allow_html=True)
            
            expiry_chart = get_expiry_chart()
            if expiry_chart:
                st.image(expiry_chart)
            else:
                st.info("Add ingredients to see expiry analytics")
                
            category_chart = get_category_distribution()
            if category_chart:
                st.image(category_chart)
            else:
                st.info("Add categorized ingredients to see distribution")
        
        with tabs[2]:
            st.markdown("<div class='pantry-title'>Settings</div>", unsafe_allow_html=True)
            st.toggle("Dark Mode", key="dark_mode")
            st.toggle("Show Expiry Alerts", value=True, key="show_expiry_alerts")
            st.selectbox("Default Search Filter", ["All", "Quick Meals", "Vegetarian", "Low Calorie"])
            st.select_slider("Recipe Complexity Level", options=["Easy", "Medium", "Advanced"])

    # Display the appropriate page based on session state
    if st.session_state.current_tab == "Find Recipes":
        display_search_page()
    else:
        display_recipe_details_page()

def display_search_page():
    # Header area with more attractive styling
    st.markdown(
        """
        <div class="header-banner">
            <h1 style='margin-bottom: 10px;'>🧙‍♂️ Kitchen Genie</h1>
            <p style='font-size: 18px;'>Transform your ingredients into delicious meals!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Main section with tabs
    tabs = st.tabs(["🔍 Recipe Search", "❤️ Favorites", "🛒 Shopping List"])
    
    with tabs[0]:
        # Search controls
        col1, col2, col3 = st.columns([2,1,1])
        
        with col1:
            st.subheader("Find recipes with your expiring ingredients")
        
        with col2:
            diet_filter = st.selectbox(
                "Dietary Preference",
                ["All", "Vegetarian", "Vegan", "Gluten-Free", "Low-Carb"],
                key="diet_filter"
            )
            
        with col3:
            meal_type = st.selectbox(
                "Meal Type",
                ["Any", "Breakfast", "Lunch", "Dinner", "Dessert", "Snack"],
                key="meal_type"
            )
        
        # Search button with better styling
        search_col1, search_col2, search_col3 = st.columns([1,2,1])
        with search_col2:
            if st.button("🔍 Find Recipes", use_container_width=True):
                search_recipes()
        
        # Display recipe cards if we have data
        if st.session_state.recipes_data:
            st.markdown("---")
            st.subheader("🥗 Matching Recipes")
            
            # Create columns for recipe cards
            cols = st.columns(3)
            for idx, match in enumerate(st.session_state.recipes_data["matches"]):
                with cols[idx % 3]:
                    is_favorite = match['metadata']['recipe_name'] in st.session_state.favorite_recipes
                    favorite_icon = "❤️" if is_favorite else "🤍"
                    
                    # Enhanced recipe card with custom CSS
                    with st.container():
                        st.markdown(f"""
                        <div class="recipe-card">
                            <h3>{match['metadata']['recipe_name']} {favorite_icon}</h3>
                            <p><strong>Similarity Score:</strong> {match['score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display ingredient tags
                        st.markdown("**Key Ingredients:**")
                        ingredients_list = match['metadata']['ingredients'].split(',')[:3]
                        ingredients_html = ' '.join([f'<span class="ingredient-tag">{ing.strip()}</span>' for ing in ingredients_list])
                        st.markdown(f"{ingredients_html} ...", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"View Recipe", key=f"view_{idx}"):
                                select_recipe(match["metadata"])
                        with col2:
                            if st.button(favorite_icon, key=f"fav_{idx}"):
                                toggle_favorite(match['metadata']['recipe_name'])
                                st.rerun()
        else:
            # Show sample recipes or tips when no search has been performed
            st.info("Click 'Find Recipes' to discover meals you can make with your pantry ingredients!")
            
            # Display recipe inspiration with more attractive styling
            st.markdown("<h3 style='text-align: center; margin: 30px 0 20px 0;'>🌟 Recipe Inspiration</h3>", unsafe_allow_html=True)
            
            inspiration_cols = st.columns(3)
            
            with inspiration_cols[0]:
                st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="text-align: center; color: #e74c3c;">Quick Weeknight Dinners</h4>
                <ul>
                <li>One-pot pasta dishes</li>
                <li>Sheet pan meals</li>
                <li>Stir-fries</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with inspiration_cols[1]:
                st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="text-align: center; color: #3498db;">Meal Prep Ideas</h4>
                <ul>
                <li>Grain bowls</li>
                <li>Batch cooking soups</li>
                <li>Freezer-friendly casseroles</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with inspiration_cols[2]:
                st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="text-align: center; color: #2ecc71;">Reduce Food Waste</h4>
                <ul>
                <li>Vegetable stock from scraps</li>
                <li>Overripe fruit smoothies</li>
                <li>Creative leftovers</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[1]:
        if not st.session_state.favorite_recipes:
            st.info("Your favorites list is empty. Heart ❤️ recipes to save them here!")
        else:
            st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Your Favorite Recipes</h3>", unsafe_allow_html=True)
            
            for idx, fav in enumerate(st.session_state.favorite_recipes):
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0;">{fav} ❤️</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("❌ Remove", key=f"remove_{idx}"):
                            st.session_state.favorite_recipes.remove(fav)
                            st.rerun()
    
    with tabs[2]:
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>🛒 Shopping List</h3>", unsafe_allow_html=True)
        
        # Shopping list management with better styling
        col1, col2 = st.columns([3, 1])
        with col1:
            shopping_item = st.text_input("Add item to shopping list")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add to List", use_container_width=True):
                if "shopping_list" not in st.session_state:
                    st.session_state.shopping_list = []
                st.session_state.shopping_list.append({"item": shopping_item, "checked": False})
        
        # Display shopping list with checkboxes and better styling
        if "shopping_list" in st.session_state and st.session_state.shopping_list:
            with st.container():
                st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h4 style="margin-bottom: 15px;">Your Shopping List</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, item in enumerate(st.session_state.shopping_list):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        checked = st.checkbox(item["item"], value=item["checked"], key=f"shop_item_{i}")
                        st.session_state.shopping_list[i]["checked"] = checked
                    with col2:
                        if st.button("🗑️", key=f"delete_item_{i}"):
                            st.session_state.shopping_list.pop(i)
                            st.rerun()
            
            # Clear completed items button with better styling
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear Completed Items", use_container_width=True):
                st.session_state.shopping_list = [item for item in st.session_state.shopping_list if not item["checked"]]
                st.rerun()
        else:
            st.info("Add items to your shopping list")

def display_recipe_details_page():
    if not st.session_state.selected_recipe:
        st.warning("No recipe selected. Please choose a recipe from the Find Recipes tab.")
        if st.button("Back to Recipe Search"):
            go_to_find_recipes()
        return

    recipe = st.session_state.selected_recipe
    
    # Toggle cooking mode - simplified view for following instructions
    cooking_mode = st.toggle("👩‍🍳 Cooking Mode", value=st.session_state.cooking_mode)
    if cooking_mode != st.session_state.cooking_mode:
        toggle_cooking_mode()
        st.rerun()
    
    # Navigation and favorites header with improved styling
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            go_to_find_recipes()
            return
    
    with col2:
        st.markdown(f"""
        <h1 style="text-align: center; margin: 0;">{recipe['recipe_name']}</h1>
        """, unsafe_allow_html=True)
    
    with col3:
        is_favorite = recipe['recipe_name'] in st.session_state.favorite_recipes
        if st.button("❤️ Favorite" if is_favorite else "🤍 Add to Favorites", use_container_width=True):
            toggle_favorite(recipe['recipe_name'])
            st.rerun()
    
    # Main recipe content
    if st.session_state.cooking_mode:
        # Simplified cooking mode UI with better styling
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #3498db;">
            <h2 style="text-align: center; color: #3498db;">👩‍🍳 Cooking Mode</h2>
            <p style="text-align: center;">Follow along with simplified instructions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate instructions using LLM if not already generated
        if 'cooking_instructions' not in st.session_state:
            with st.spinner("Preparing detailed cooking instructions..."):
                prompt = ChatPromptTemplate.from_template(
                    """Create detailed step-by-step instructions for this recipe:
                    Name: {name}
                    Ingredients: {ingredients}
                    Original Directions: {directions}
                    
                    Format as a numbered list with clear, concise steps. Include timing information and visual cues for doneness.
                    """
                )
                llm = ChatGroq(model_name="Llama3-8b-8192", api_key=GROQ_API_
