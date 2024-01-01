
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, FloatField
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    protocol_type = StringField('PROTOCOL TYPE', validators=[DataRequired()])
    flag = StringField('FLAG', validators=[DataRequired()])
    src_bytes = IntegerField('SOURCE BYTES')
    dst_bytes = IntegerField('DESTINATION BYTES')
    hot = IntegerField('HOT')
    count = IntegerField('COUNT')
    srv_count = IntegerField('SERVICE COUNT')
    same_srv_rate = FloatField('SAME SERVICE RATE')
    dst_host_count = IntegerField('DESTINATION HOST COUNT')
    dst_host_srv_count = IntegerField('DESTINATION HOST SERVICE COUNT')
    dst_host_same_srv_rate = FloatField('DESTINATION HOST SAME SERVICE RATE')
    dst_host_diff_srv_rate = FloatField('DESTINATION HOST DIFF SERVICE RATE')
    dst_host_same_src_port_rate = FloatField('DESTINATION HOST SAME SERVICE PORT RATE')
    dst_host_rerror_rate = FloatField('DESTINATION HOST RERROR RATE')
    submit = SubmitField('Predict')