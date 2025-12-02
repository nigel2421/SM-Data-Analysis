{ pkgs, ... }: {
  channel = "stable-24.05";
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
  ];

  idx = {
    previews = {
      enable = true;
      previews = {
        web = {
          command = [ "sh" "-c" ". venv/bin/activate && streamlit run app.py --server.address 0.0.0.0 --server.enableCORS=false --server.headless=true" ];
          manager = "web";
          env = { 
            STREAMLIT_SERVER_PORT = "$PORT";
          };
        };
      };
    };

    workspace = {
      onCreate = {
        setup = "python -m venv venv && . venv/bin/activate && pip install -r requirements.txt";
      };
    };
  };
}
