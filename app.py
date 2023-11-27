import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from scipy.optimize import minimize
from itertools import combinations

########################################################################################
# Functions
########################################################################################
def transform_points(points, scale, rotation, translation):
    # Apply rotation and scaling
    transform_matrix = np.array([
        [np.cos(rotation) * scale, -np.sin(rotation) * scale],
        [np.sin(rotation) * scale,  np.cos(rotation) * scale]
    ])
    transformed = np.dot(points, transform_matrix.T) + translation
    return transformed
    
def perform_adjustment(points1, points2):
    initial_guess = [1, 0, 0, 0]  # Initial scale, rotation, translation x, translation y
    result = minimize(least_squares_transformation, initial_guess, args=(points1, points2))
    scale, rotation, tx, ty = result.x
    transformed_points = transform_points(points1, scale, rotation, [tx, ty])

    # Generating a report or output
    return transformed_points, (scale, rotation, tx, ty)

def least_squares_transformation(params, points1, points2):
    scale, rotation, tx, ty = params
    transformed_points = transform_points(points1, scale, rotation, [tx, ty])
    # Compute the sum of squared differences
    return np.sum((transformed_points - points2) ** 2)


def load_data(file):
    return pd.read_csv(file)

def plot_data(df1, df2, df3):
    fig, ax = plt.subplots()
    ax.scatter(df1['easting'], df1['northing'], color='blue', label='Field')
    ax.scatter(df3['easting'], df3['northing'], color='green', label='Adjusted')
    ax.scatter(df2['easting'], df2['northing'], color='red', label='Control')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.legend()
    return fig

def combine_data(df1, df2):
    combined = pd.concat([df1, df2])
    return combined

def to_csv(df):
    output = StringIO()
    df.to_csv(output, index=False, header=False)
    return output.getvalue()

def validate_points_dataframe(df):
    if "northing" not in df.columns:
        raise "northing column required"
    if "easting" not in df.columns:
        raise "easting column required"
    if "pt_number" not in df.columns:
        raise "pt_number column required"


########################################################################################
# UI
########################################################################################

st.title('Adjust Field GCPs to Project Control Coordinate System')

st.sidebar.header('Upload CSV Files')
st.sidebar.header('each file must include pt_number,northing,easting in header row')
field_file = st.sidebar.file_uploader("Choose Field CSV file", type=['csv'])
control_file = st.sidebar.file_uploader("Choose Control CSV file", type=['csv'])

if field_file and control_file:
    field_data = load_data(field_file)
    control_data = load_data(control_file)

    validate_points_dataframe(field_data)
    validate_points_dataframe(control_data)

    control_points=control_data[["northing","easting"]].values
    field_data_filtered=field_data[field_data.pt_number.isin (control_data.pt_number)]
    field_points=field_data_filtered[["northing","easting"]].values
    
    # do first transformation to find outliers
    transformed_points, transformation_params = perform_adjustment(field_points, control_points)
    outlier_detection = field_data_filtered
    outlier_detection[["northing","easting"]] = transformed_points
    outlier_detection["error"] = np.sqrt((outlier_detection["northing"]-control_data["northing"])**2 + (outlier_detection["easting"]-control_data["easting"])**2)
    outlier_detection = outlier_detection.sort_values(by='error', ascending=False)
    num_to_remove = int(len(outlier_detection) * 0.26)
    outlier_detection = outlier_detection.iloc[:num_to_remove]
    control_data["dropped"] = control_data.pt_number.isin(outlier_detection.pt_number)

    # filter the outliers from the control and filter field data
    control_data_filtered = control_data[~control_data.pt_number.isin(outlier_detection.pt_number)]
    field_data_filtered = field_data_filtered[~field_data_filtered.pt_number.isin(outlier_detection.pt_number)]
    
    control_points=control_data_filtered[["northing","easting"]].values
    field_points=field_data_filtered[["northing","easting"]].values
    transformed_points, transformation_params = perform_adjustment(field_points, control_points)
    tp=transform_points(field_data[["northing","easting"]],transformation_params[0],
                        transformation_params[1],[transformation_params[2],transformation_params[3]])
    
    # Create a copy of field_data for transformation
    adjusted_data = field_data.copy()
    adjusted_data[["northing", "easting"]] = tp

    st.write("Control Data")
    st.dataframe(control_data,hide_index=True,use_container_width=True)
    
    st.write("Field Data matches")
    st.dataframe(field_data_filtered,hide_index=True,use_container_width=True)

    st.write("Adjusted Data")
    adjusted_data_filtered=adjusted_data[adjusted_data.pt_number.isin (field_data.pt_number)]
    st.dataframe(adjusted_data_filtered,hide_index=True,use_container_width=True)
    st.write ("Scale:")
    st.write (transformation_params[0])
    st.write ("Rotation:")
    st.write (transformation_params[1])
    st.write ("Translation:")
    st.write ([transformation_params[2],transformation_params[3]])
    
    st.write("Scatter Plot")
    fig = plot_data(field_data, control_data, adjusted_data)
    st.pyplot(fig)

    csv = to_csv(adjusted_data)
    st.download_button(label="Download adjusted CSV",
                       data=csv,
                       file_name='adjusted.csv',
                       mime='text/csv')
else:
    st.warning('Upload both field and control CSV files')
