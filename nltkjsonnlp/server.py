from nltkjsonnlp import NltkPipeline
from pyjsonnlp.microservices.flask_server import FlaskMicroservice

app = FlaskMicroservice(__name__, NltkPipeline(), base_route='/')
app.with_constituents = False
app.with_coreferences = False
app.with_dependencies = False
app.with_expressions = False

if __name__ == "__main__":
    app.run(debug=True, port=5003)
