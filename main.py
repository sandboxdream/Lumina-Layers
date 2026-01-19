"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LUMINA STUDIO v1.3                                   ║
║                    Multi-Material 3D Print Color System                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Author: [MIN]                                                                ║
║  License: CC BY-NC-SA 4.0                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Main Entry Point
"""

from ui import create_app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        inbrowser=True,
        server_port=7860,
        share=False,
        show_error=True
    )
