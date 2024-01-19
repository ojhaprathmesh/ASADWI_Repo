from flask import Flask
from flask import render_template
from flask import request, jsonify
from Interface import water_quality_check, geology_check, lithology

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('main_new.html')


@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    inp_place = user_input
    a = water_quality_check('./ai_models/Water_Quality_Model.pkl', 'data/water_quality_data.csv', place=inp_place)
    b = geology_check('./ai_models/Geology_Model.pkl', 'data/Geology Maharashtra.xlsx', place=inp_place)
    c = lithology('./ai_models/Lithology_Model.pkl', 'data/Lithology Bihar.xlsx', place=inp_place)

    # Create a dictionary to store the results
    result_dict = {
        "water_quality_result": a,
        "geology_result": b,
        "lithology_result": c
    }

    # Return the results as JSON
    return jsonify(result_dict)


if __name__ == '__main__':
    app.run(debug=True)
