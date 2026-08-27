"""
KV4P-Desktop — Python desktop app for the KV4P-HT ham radio board.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication

from kv4p_ht.app import MainWindow


def main():
    parser = argparse.ArgumentParser(description="KV4P-Desktop — Ham Radio App")
    parser.add_argument('--version', action='version', version='KV4P-Desktop v0.1.4')
    parser.add_argument('--callsign', type=str, help="Default callsign")
    parser.add_argument('--freq', type=float, help="Initial RX frequency in MHz")
    parser.add_argument('--offset', type=float, help="Repeater offset in MHz")
    parser.add_argument('--rigctl-host', type=str, help="Hamlib rigctld host")
    parser.add_argument('--rigctl-port', type=int, help="Hamlib rigctld port")
    parser.add_argument('--igate', action='store_true', help="Start APRS iGate on launch")
    parser.add_argument('--beacon', action='store_true', help="Start APRS beacon on launch")
    parser.add_argument('--scanner', action='store_true', help="Start frequency scanner on launch")
    parser.add_argument('--log-level', type=str, choices=['debug', 'info', 'warn', 'error'],
                        default='info', help="Logging level")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("KV4P-Desktop")
    app.setOrganizationName("kv4p")

    window = MainWindow()

    if args.callsign:
        window.callsign = args.callsign.upper()
        window._callsign_edit.setText(window.callsign)

    if args.freq is not None:
        window._freq_edit.setText(f"{args.freq:.3f}")
        window._set_frequency()

    if args.offset is not None:
        window._offset_edit.setText(f"{args.offset:.3f}")
        window._set_frequency()

    if args.rigctl_host:
        window._rigctl_host.setText(args.rigctl_host)
        if args.rigctl_port:
            window._rigctl_port.setText(str(args.rigctl_port))
        window._rigctl_btn.setChecked(True)
        window._toggle_rigctld(True)

    if args.igate:
        window._igate_btn.setChecked(True)
        window._toggle_igate(True)

    if args.beacon:
        window._beacon_btn.setChecked(True)
        window._toggle_beacon(True)

    if args.scanner:
        window._scan_btn.click()

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
