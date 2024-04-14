import requests
import json
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, JsCode, ColumnsAutoSizeMode, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


st.set_page_config(
    page_title="Арбитраж",
    page_icon="🎓",
    initial_sidebar_state="expanded",
    layout="wide",
)

with open("app/css/AG_GRID_LOCALE_RU.txt", "r") as f:
    AG_CRID_LOCALE_RU = json.load(f)


def plot_change_table(df, key):
    js = JsCode(
        """
            function(e) {
                let api = e.api;     
                let sel = api.getSelectedRows();
                api.applyTransaction({remove: sel});
            };
            """
    )

    gd = GridOptionsBuilder.from_dataframe(df, columns_auto_size_mode=0)
    gd.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=5
    )
    gd.configure_grid_options(stopEditingWhenCellsLoseFocus=True, rowHeight=600)  # , rowHeight=80
    gd.configure_grid_options(localeText=AG_CRID_LOCALE_RU)
    gd.configure_default_column(editable=True, groupable=True, wrapText = True)
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
    gd.configure_grid_options(onRowSelected=js, pre_selected_rows=[])
    gridoptions = gd.build()
    grid_table = AgGrid(
        df,
        gridOptions=gridoptions,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        theme="alpine",
        key=key,
        height=1000,
    )

    return grid_table


styles = {
    "container": {"padding": "0!important", "background-color": "#fafafa"},
    "icon": {
        "color": "#db0404",
        "font-size": "20px",
        "--hover-color": "#cccccc",
    },
    "nav-link": {
        "color": "black",
        "font-size": "15px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#cccccc",
    },
    "nav-link-selected": {
        "color": "#ffffff",
        "background-color": "#808080",
    },
}


with st.sidebar:
    st.write("##")
    selected = option_menu(
        None,
        [
            "Классификатор",
            "Архив",
        ],
        icons=[
            "list-task",
            "database",
        ],
        menu_icon="cast",
        default_index=0,
        styles=styles,
    )

if selected == "Классификатор":

    col1, col2, col3 = st.columns((1, 3, 1))

    with col2:

        st.write(
            """
    # 📃 Классификатор докуметов

    принимает на вход файлы расширения **.DOC**, **.RTF**, **.PDF**, **.DOCX**

    """
        )

        st.write("**Загрузка документа**")

        uploaded_file = st.file_uploader(
            label="Выберите файл:",
            type=["doc", "docx", "pdf", "rtf"],
            accept_multiple_files=False,
            help="укажите путь к файлу или перетащите его в онко загрузки",
        )
        if uploaded_file:
            url = "http://localhost:8080/docs.add"
            payload = {"filename": uploaded_file.name}
            response = requests.post(
                url, data=payload, files={"uploaded_file": uploaded_file.getvalue()}
            )
            contract_name = uploaded_file.name
            st.success("Файл успешно загружен", icon="✅")
            data = response.json()["data"]
            try:
                content = data["content"]
            

                st.components.v1.html(
                    content, width=None, height=300, scrolling=True
                )

                st.success("Вид документа успешно определён", icon="✅")

                st.write(
                    f"""**Результаты:**
                    
                📌 Наименование файла: {contract_name}
                
        ✔️ Предсказанный вид документа:  {data["label"]}

                    """
                )
            except:
                st.error("Такой документ уже есть в базе данных", icon="❌")
        else:
            st.error("Вы ничего не вабрали", icon="❌")
            



if selected == "Архив":

    st.write(
        """
# 📃 Архив классифицированных докуметов

**.DOC**, **.RTF**, **.PDF**, **.DOCX**

"""
    )
    url = "http://localhost:8080/docs.list"

    response = requests.get(url)

    files = response.json()["data"]["files"]

    if len(files) == 0:

        st.error("В базе данных пока нет ни каких документов", icon="❌")

    else:

        df = pd.DataFrame(files)

        aggrid_files = plot_change_table(df, key="abc")


st.markdown(
    "<h5 style='text-align: center; color: blac;'> ©️ Команда ЛИФТ </h5>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h5 style='text-align: center; color: blac;'> Цифровой проры 2024, Сочи </h5>",
    unsafe_allow_html=True,
)
