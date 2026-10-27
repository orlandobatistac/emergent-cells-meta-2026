# What the user needs to enable the real Meta call

The code is ready to call Meta using the documented endpoint, but **credentials are not configured in the running Windows process**.

1. In the local developer portal, create/retrieve your Model API key after verifying billing and whether hackathon credits are active.
2. Set MODEL_API_KEY as a private operating-system environment variable for the process that launches the laboratory. Do not save the value in Git, paste it into chat or put it into a browser textbox.
3. Install optional client package with: python -m pip install -e ".[meta]"
4. Run: python -m emergent_cells.demo.server
5. Choose Meta Muse Spark in the challenge panel, submit one description, and check the response source says meta. The app rejects output outside safe WorldSpec bounds, so some Meta proposals can fail validation.

You can always select the visibly labeled offline template to test the complete world-generation-to-simulation pipeline without API charges.

No claim of an actual Meta API request or granted credits is made at this stage.
