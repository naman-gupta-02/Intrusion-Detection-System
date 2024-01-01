from flask import Flask, render_template, flash, redirect, url_for
from forms import PredictionForm
from System.codeCNN import get_accuracy, predict_user_input

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

accuracy = get_accuracy()
accuracy = accuracy * 100
accuracy = "{:.2f}".format(accuracy)

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = PredictionForm()
    if form.validate_on_submit():
        user_input_example = {
            'protocol_type' : form.protocol_type.data,
            'flag' : form.flag.data,
            'src_bytes' : form.src_bytes.data,
            'dst_bytes' : form.dst_bytes.data,
            'hot' : form.hot.data,
            'count' : form.count.data,
            'srv_count' : form.srv_count.data,
            'same_srv_rate' : form.same_srv_rate.data,
            'dst_host_count' : form.dst_host_count.data,
            'dst_host_srv_count' : form.dst_host_srv_count.data,
            'dst_host_same_srv_rate' : form.dst_host_same_srv_rate.data,
            'dst_host_diff_srv_rate' : form.dst_host_diff_srv_rate.data,
            'dst_host_same_src_port_rate' : form.dst_host_same_src_port_rate.data,
            'dst_host_rerror_rate' : form.dst_host_rerror_rate.data
        }
        predicted_class = predict_user_input(user_input_example)
        # print(predicted_class)
        if(predicted_class != [0]): 
            flash('No intrusion detected in the network!', 'success')
        else:
            flash('An anomaly is detected in the network!', 'danger')
        return redirect(url_for('home'))
    # if request.method == 'POST' and not form.validate():
    #     print(form.errors)
    return render_template('values.html',  title= 'Prediction', form= form, accuracy = accuracy)

if __name__ == "__main__":
    app.run(debug=True)