import streamlit as st
import streamlit.components.v1 as components
import numpy as np

from modules.loader import molecule_to_xyz

# ============================================================
# INTERACTIVE 3D MOLECULE VIEWER
# ============================================================

def display_3d_molecule(
    xyz,
    height=420,
    representation="Stick + Sphere",
    show_labels=True,
    label_color="yellow",
    atom_scale=0.30,
    bond_radius=0.12,
    background="white",
    spin=False,
    show_axes=False,
    show_controls=True,
    projection="free",
    orthographic=False,
    selected_atom_index=None,
    enable_atom_selection=False,
    atom_click_bridge_label=None,
    download_filename=None,
    download_filename_xyz=None,
):
    """
    Display an interactive 3D molecule using 3Dmol.js.

    Parameters
    ----------
    xyz : str
        Molecule in XYZ format.

    height : int
        Total viewer height in pixels.

    representation : str
        Initial molecular representation.

        Options:
            "Stick + Sphere"
            "Stick"
            "Sphere"
            "Line"

    show_labels : bool
        Show atomic element labels.

    label_color : str
        Initial label color.

    atom_scale : float
        Sphere scale for atoms.

    bond_radius : float
        Radius of stick representation.

    background : str
        Initial viewer background.

        Options:
            "black"
            "white"
            "transparent"

    spin : bool
        Enable automatic rotation initially.

    show_axes : bool
        Show Cartesian X/Y/Z axes initially.

    show_controls : bool
        Display viewer controls and export buttons.

    projection : str
        Initial molecular projection.

        Options:
            "free"
            "x"
            "y"
            "z"

    orthographic : bool
        Use orthographic projection instead of perspective.

    selected_atom_index : int or None
        0-based index (matching source array order -- see
        enable_atom_selection below) of an atom to highlight on
        load. None means no atom is highlighted.

    enable_atom_selection : bool
        Make atoms clickable. On click, the atom is highlighted in
        the viewer and its 0-based index is reported back to
        Streamlit through the hidden bridge widget identified by
        atom_click_bridge_label (see that parameter). This index is
        the atom's 3Dmol.js `.serial` value, which -- verified
        directly against 3Dmol.js's XYZ parser source -- is assigned
        in strict, sequential file-line order starting at 0 for a
        single (non-multimodel) XYZ load, exactly matching
        molecule_to_xyz()'s own write order (a plain zip() over the
        source atNUM/atXYZ arrays with no reordering). So
        atom.serial == the row index into atNUM/atXYZ/hCHG/atPOL/
        vdwR for the same molecule.

    atom_click_bridge_label : str or None
        The exact `label` text of a hidden Streamlit
        st.number_input widget (rendered by the caller, elsewhere in
        the Streamlit script -- not by this function) used to carry
        a clicked atom's index from this component's JS back into
        Python. Required when enable_atom_selection=True. See the
        "ATOM CLICK BRIDGE" section below for exactly how this
        works and its limitations.

    download_filename : str or None
        Filename offered when the viewer's always-present "Download
        3D view" button is used. Defaults to "molecule_3d.png" when
        not given. This is a client-side-only PNG export (via
        3Dmol.js's own viewer.pngURI(), reflecting whatever is
        currently on screen -- rotation/zoom/representation/
        background included) with no Python round-trip, unlike the
        atom-click bridge above -- it needed none of that
        machinery's fragility.

    download_filename_xyz : str or None
        Filename offered by the always-present "Download XYZ"
        button. Defaults to "molecule_3d.xyz" when not given. Same
        pattern as display_comparative_3d_viewers()'s own XYZ
        button: a Blob() of the exact XYZ text already used to
        build this viewer's model, downloaded via a temporary
        <a download> element -- also client-side-only, no Python
        round-trip.
    """

    # ========================================================
    # ESCAPE XYZ FOR JAVASCRIPT
    # ========================================================

    xyz_js = (
        xyz
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    # ========================================================
    # 3D VIEW DOWNLOAD FILENAME
    # ========================================================

    download_filename_js = (
        str(download_filename)
        if download_filename
        else "molecule_3d.png"
    ).replace("\\", "\\\\").replace('"', '\\"')

    # Same "molecule_<id>.xyz" convention (no "_3d" suffix) already
    # used by display_comparative_3d_viewers()'s own XYZ button.
    download_filename_xyz_js = (
        str(download_filename_xyz)
        if download_filename_xyz
        else "molecule_3d.xyz"
    ).replace("\\", "\\\\").replace('"', '\\"')

    # ========================================================
    # BACKGROUND
    # ========================================================

    background_map = {
        "black": "#000000",
        "white": "#FFFFFF",
        "transparent": "rgba(0,0,0,0)",
    }

    initial_background = background_map.get(
        background,
        "#000000",
    )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    if representation == "Stick + Sphere":

        style_js = f"""
        {{
            stick: {{
                radius: {bond_radius}
            }},
            sphere: {{
                scale: {atom_scale}
            }}
        }}
        """

    elif representation == "Stick":

        style_js = f"""
        {{
            stick: {{
                radius: {bond_radius}
            }}
        }}
        """

    elif representation == "Sphere":

        style_js = f"""
        {{
            sphere: {{
                scale: {atom_scale}
            }}
        }}
        """

    elif representation == "Line":

        style_js = """
        {
            line: {}
        }
        """

    else:

        style_js = f"""
        {{
            stick: {{
                radius: {bond_radius}
            }},
            sphere: {{
                scale: {atom_scale}
            }}
        }}
        """

    # ========================================================
    # INITIAL LABELS
    # ========================================================
    #
    # `labelsEnabled` (a JS variable, not the removed `labels_js`
    # string this used to build) is the single source of truth for
    # whether atom labels are shown, seeded directly from this
    # Python `show_labels` argument. updateLabels() reads/writes it.
    #
    # Root cause of the previous bug: this block used to build a
    # `labels_js` string (via viewer.addPropertyLabels(...)) that
    # was NEVER interpolated into the HTML template below -- dead
    # code. The only thing that ever actually drew labels was
    # updateLabels() (see that function for the rest of the fix),
    # which read the *DOM* checkbox's `.checked` property as its
    # only source of truth. The one live call site of this function
    # renders with show_controls=False, so that checkbox element
    # never exists in the DOM -- updateLabels() always hit its
    # "!labelsCheck" early-return branch and labels never appeared,
    # regardless of what `show_labels` was set to.

    initial_labels_enabled_js = (
        "true"
        if show_labels
        else "false"
    )

    # ========================================================
    # INITIAL SPIN
    # ========================================================

    if spin:

        spin_js = """
        viewer.spin("y", 0.5);
        """

    else:

        spin_js = ""

    # ========================================================
    # AXES
    # ========================================================

    axes_js = """
    let axisObjects = [];

    function clearAxes() {

        axisObjects.forEach(
            function(obj) {
                viewer.removeShape(obj);
            }
        );

        axisObjects = [];
    }


    function drawAxes() {

        clearAxes();

        const extent = 2.0;

        const axisDefinitions = [

            {
                start: {x: 0, y: 0, z: 0},
                end:   {x: extent, y: 0, z: 0},
                color: "red",
                label: "X"
            },

            {
                start: {x: 0, y: 0, z: 0},
                end:   {x: 0, y: extent, z: 0},
                color: "green",
                label: "Y"
            },

            {
                start: {x: 0, y: 0, z: 0},
                end:   {x: 0, y: 0, z: extent},
                color: "blue",
                label: "Z"
            }

        ];


        axisDefinitions.forEach(
            function(axis) {

                const line =
                    viewer.addLine({

                        start: axis.start,
                        end: axis.end,
                        color: axis.color,
                        linewidth: 2

                    });


                axisObjects.push(line);


                const label =
                    viewer.addLabel(
                        axis.label,
                        {

                            position: axis.end,
                            fontColor: axis.color,
                            fontSize: 14,
                            backgroundOpacity: 0,
                            showBackground: false,
                            inFront: true

                        }
                    );


                axisObjects.push(label);

            }
        );
    }
    """

    # ========================================================
    # ATOM CLICK BRIDGE
    # ========================================================
    #
    # components.html() renders a one-way iframe: there is no
    # built-in return channel to Python, and this codebase has no
    # bidirectional custom component anywhere (checked: no
    # components.declare_component call and no postMessage bridge
    # in modules/ or pages/). The mechanism below is the smallest
    # viable workaround: reach into the *parent* document (same
    # origin -- Streamlit's own iframe sandbox for components.html
    # includes "allow-same-origin", confirmed from the installed
    # Streamlit frontend bundle) and drive a hidden native
    # st.number_input's <input> element exactly the way a real user
    # interaction would (focus, set value via the native property
    # setter so React's controlled-input listener actually sees the
    # change, dispatch input/change, blur to commit) -- then
    # Streamlit's own existing widget -> backend sync does the rest.
    #
    # Real limitations of this approach (documented, not oversold):
    #   - It costs an actual Streamlit rerun (a brief "Running..."),
    #     not an instant in-place update.
    #   - It depends on Streamlit continuing to render
    #     st.number_input as a plain <input> whose aria-label equals
    #     its `label` text (true as of the Streamlit version this
    #     was built against; not a guaranteed public contract).
    #   - It depends on the iframe sandbox continuing to include
    #     allow-same-origin.
    #   - It can report exactly one value (this atom index) -- it is
    #     not a general-purpose channel.
    #
    # The atom index reported is 3Dmol.js's own atom.serial. This
    # was verified directly against 3Dmol.js's XYZ parser source
    # (not assumed): for a single/non-multimodel XYZ load, the
    # parser assigns serial = f for f running sequentially from 0 in
    # strict file-line order. molecule_to_xyz() (modules/loader.py)
    # writes atom lines via a plain zip(atNUM, atXYZ) with no
    # reordering. So atom.serial == the row index into
    # atNUM/atXYZ/hCHG/atPOL/vdwR for this same molecule.

    if enable_atom_selection:

        bridge_label_js = (
            str(atom_click_bridge_label)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        if selected_atom_index is not None:

            atom_initial_highlight_js = f"""
            (function() {{
                const initialModel = viewer.getModel();
                if (initialModel) {{
                    const initialAtoms =
                        initialModel.selectedAtoms(
                            {{ serial: {int(selected_atom_index)} }}
                        );
                    if (initialAtoms.length > 0) {{
                        qm7xHighlightAtom(initialAtoms[0]);
                    }}
                }}
            }})();
            """

        else:

            atom_initial_highlight_js = ""

        atom_selection_js = f"""
        let qm7xHighlightShape = null;

        // Same per-element van der Waals radii 3Dmol.js itself uses
        // to size sphere/stick+sphere atoms (GLModel.vdwRadii) --
        // duplicated here (read-only lookup, not reused by
        // reference) so the highlight halo can be sized as a
        // multiple of THIS atom's own rendered radius rather than a
        // fixed absolute size. A fixed radius (0.45, the previous
        // value) reads as an obvious ring around a small atom like
        // H (vdW 1.20 * a typical 0.30 atom_scale =~ 0.36) but gets
        // swallowed by or barely pokes out of a much larger atom's
        // own sphere, e.g. S (vdW 1.80) or I (vdW 1.98) at the same
        // scale -- which is exactly the reported "hard to notice on
        // larger atoms" symptom.
        const qm7xVdwRadii = {{
            H: 1.20, He: 1.40, Li: 1.82, Be: 1.53, B: 1.92,
            C: 1.70, N: 1.55, O: 1.52, F: 1.47, Ne: 1.54,
            Na: 2.27, Mg: 1.73, Al: 1.84, Si: 2.10, P: 1.80,
            S: 1.80, Cl: 1.75, Ar: 1.88, K: 2.75, Ca: 2.31,
            Br: 1.85, I: 1.98
        }};

        const qm7xAtomScale = {atom_scale};

        function qm7xHighlightAtom(atom) {{

            if (qm7xHighlightShape) {{
                viewer.removeShape(qm7xHighlightShape);
                qm7xHighlightShape = null;
            }}

            if (!atom) {{
                viewer.render();
                return;
            }}

            const baseRadius =
                (qm7xVdwRadii[atom.elem] || 1.5) * qm7xAtomScale;

            // Always noticeably bigger than the atom's own sphere
            // (1.6x its radius) plus a small additive floor so the
            // halo still reads clearly in Stick/Line representations
            // where atom_scale isn't actually rendered as a visible
            // sphere at all.
            const haloRadius = (baseRadius * 1.2) + 0.10;

            qm7xHighlightShape = viewer.addSphere({{
                center: {{ x: atom.x, y: atom.y, z: atom.z }},
                radius: haloRadius,
                color: "cyan",
                wireframe: true,
                linewidth: 2
            }});

            viewer.render();
        }}

        function qm7xSendClickedAtomToStreamlit(atomIndex) {{

            try {{

                const parentDoc = window.parent.document;

                const bridgeInput = parentDoc.querySelector(
                    'input[aria-label="{bridge_label_js}"]'
                );

                if (!bridgeInput) {{
                    console.warn(
                        "qm7x atom-click bridge: input not found " +
                        "in parent document."
                    );
                    return;
                }}

                const nativeSetter =
                    Object.getOwnPropertyDescriptor(
                        window.parent.HTMLInputElement.prototype,
                        "value"
                    ).set;

                bridgeInput.focus();
                nativeSetter.call(bridgeInput, String(atomIndex));
                bridgeInput.dispatchEvent(
                    new Event("input", {{ bubbles: true }})
                );
                bridgeInput.dispatchEvent(
                    new Event("change", {{ bubbles: true }})
                );
                bridgeInput.blur();

            }} catch (err) {{
                console.warn(
                    "qm7x atom-click bridge failed:",
                    err
                );
            }}
        }}

        // Tracks whether the atom-specific callback below already
        // handled the CURRENT click, so the empty-space listener
        // (right after) knows whether to also clear the selection.
        // 3Dmol.js's setClickable() callback only fires when an
        // atom is actually hit -- there is no built-in "clicked
        // nothing" callback, and no public pickAtom()/hit-test API
        // in this bundled version (checked). This flag +
        // deferred-check pattern is the standard workaround: both
        // this per-atom callback and the container-level listener
        // below fire from the SAME underlying browser click, so
        // checking the flag on the next tick (setTimeout 0, after
        // both handlers for this click have already run) reliably
        // tells the two cases apart regardless of 3Dmol.js's
        // internal event order.
        let qm7xAtomClickHandled = false;

        viewer.setClickable(
            {{}},
            true,
            function(atom, viewerRef, event, container) {{
                qm7xAtomClickHandled = true;
                qm7xHighlightAtom(atom);
                qm7xSendClickedAtomToStreamlit(atom.serial);
            }}
        );

        // ==================================================
        // CLEAR SELECTION ON EMPTY-SPACE CLICK
        // ==================================================
        //
        // Capture phase (the `true` third argument) so this always
        // runs for every click on the viewer, hit or miss, even if
        // 3Dmol.js's own bubble-phase handling ever stops
        // propagation on a hit.
        const qm7xViewerEl = document.getElementById("viewer");

        if (qm7xViewerEl) {{

            qm7xViewerEl.addEventListener(
                "click",
                function() {{

                    setTimeout(
                        function() {{

                            if (!qm7xAtomClickHandled) {{

                                // No atom callback fired for this
                                // click -- empty space. Clear the
                                // highlight and report -1 through
                                // the SAME bridge used everywhere
                                // else (reset/molecule change), so
                                // clicked_atom_index resolves to
                                // None exactly like today.
                                qm7xHighlightAtom(null);
                                qm7xSendClickedAtomToStreamlit(-1);
                            }}

                            qm7xAtomClickHandled = false;
                        }},
                        0
                    );
                }},
                true
            );
        }}

        {atom_initial_highlight_js}
        """

    else:

        atom_selection_js = ""

    # ========================================================
    # CONTROLS HTML
    # ========================================================

    if show_controls:

        controls_html = f"""
        <div id="controls">

            <div class="control-row">

                <label>

                    Background

                    <select id="backgroundSelect">

                        <option value="black"
                            {"selected" if background == "black" else ""}>
                            Black
                        </option>

                        <option value="white"
                            {"selected" if background == "white" else ""}>
                            White
                        </option>

                        <option value="transparent"
                            {"selected" if background == "transparent" else ""}>
                            Transparent
                        </option>

                    </select>

                </label>


                <label>

                    Representation

                    <select id="representationSelect">

                        <option value="Stick + Sphere"
                            {"selected" if representation == "Stick + Sphere" else ""}>
                            Stick + Sphere
                        </option>

                        <option value="Stick"
                            {"selected" if representation == "Stick" else ""}>
                            Stick
                        </option>

                        <option value="Sphere"
                            {"selected" if representation == "Sphere" else ""}>
                            Sphere
                        </option>

                        <option value="Line"
                            {"selected" if representation == "Line" else ""}>
                            Line
                        </option>

                    </select>

                </label>


                <label>

                    Atom size

                    <input
                        id="atomScale"
                        type="range"
                        min="0.15"
                        max="0.60"
                        step="0.01"
                        value="{atom_scale}"
                    >

                </label>


                <label>

                    Bond radius

                    <input
                        id="bondRadius"
                        type="range"
                        min="0.03"
                        max="0.30"
                        step="0.01"
                        value="{bond_radius}"
                    >

                </label>

            </div>


            <div class="control-row">

                <label class="checkbox-label">

                    <input
                        id="labelsCheck"
                        type="checkbox"
                        {"checked" if show_labels else ""}
                    >

                    Atom labels

                </label>


                <label>

                    Projection

                    <select id="projectionSelect">

                        <option value="free"
                            {"selected" if projection == "free" else ""}>
                            Free 3D
                        </option>

                        <option value="x"
                            {"selected" if projection == "x" else ""}>
                            X axis
                        </option>

                        <option value="y"
                            {"selected" if projection == "y" else ""}>
                            Y axis
                        </option>

                        <option value="z"
                            {"selected" if projection == "z" else ""}>
                            Z axis
                        </option>

                    </select>

                </label>


                <label class="checkbox-label">

                    <input
                        id="orthographicCheck"
                        type="checkbox"
                        {"checked" if orthographic else ""}
                    >

                    Orthographic

                </label>


                <label class="checkbox-label">

                    <input
                        id="spinCheck"
                        type="checkbox"
                        {"checked" if spin else ""}
                    >

                    Auto rotate

                </label>


                <button id="resetButton">
                    ↻ Reset
                </button>


                <button id="pngButton">
                    ↓ PNG
                </button>


                <button id="xyzButton">
                    ↓ XYZ
                </button>

            </div>

        </div>
        """

    else:

        controls_html = ""

    # ========================================================
    # HTML
    # ========================================================

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="UTF-8">

        <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>

        <style>

            html,
            body {{

                margin: 0;
                padding: 0;

                width: 100%;
                height: 100%;

                overflow: hidden;

                background: transparent;

                font-family:
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    sans-serif;
            }}


            #viewerContainer {{

                position: relative;

                width: 100%;
                height: 100%;

                background: transparent;

                overflow: hidden;
            }}


            #viewer {{

                position: absolute;

                left: 0;
                right: 0;
                bottom: 0;
                top: 0;

                width: 100%;
                height: 100%;
            }}


            .qm7xDownloadBtn {{

                position: absolute;

                top: 8px;

                z-index: 200;

                padding: 4px 10px;

                font-family: -apple-system, BlinkMacSystemFont,
                    sans-serif;
                font-size: 12px;

                color: #E2E8F0;
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(226, 232, 240, 0.35);
                border-radius: 5px;

                cursor: pointer;
            }}

            .qm7xDownloadBtn:hover {{

                background: rgba(15, 23, 42, 0.85);
            }}

            #qm7xDownload3DBtn {{
                right: 8px;
            }}

            #qm7xDownloadXYZBtn {{
                right: 68px;
            }}


            #controls {{

                position: absolute;

                top: 8px;
                left: 8px;
                right: 8px;

                z-index: 100;

                padding: 8px 10px;

                border-radius: 8px;

                background:
                    rgba(15, 23, 42, 0.92);

                border:
                    1px solid
                    rgba(148, 163, 184, 0.25);

                color: #F8FAFC;

                backdrop-filter: blur(8px);

                font-size: 11px;
            }}


            .control-row {{

                display: flex;

                align-items: center;

                flex-wrap: wrap;

                gap: 8px;

                margin-bottom: 5px;
            }}


            .control-row:last-child {{

                margin-bottom: 0;
            }}


            label {{

                display: flex;

                align-items: center;

                gap: 5px;

                white-space: nowrap;
            }}


            select {{

                background: #111827;

                color: #F8FAFC;

                border:
                    1px solid #334155;

                border-radius: 5px;

                padding: 3px 5px;

                font-size: 11px;
            }}


            input[type="range"] {{

                width: 70px;
            }}


            input[type="checkbox"] {{

                accent-color: #19DBDE;
            }}


            button {{

                background: #111827;

                color: #F8FAFC;

                border:
                    1px solid #334155;

                border-radius: 5px;

                padding: 4px 8px;

                font-size: 11px;

                cursor: pointer;
            }}


            button:hover {{

                background: #1E293B;

                border-color: #19DBDE;
            }}


            .checkbox-label {{

                padding: 2px 4px;
            }}

        </style>

    </head>


    <body>

        <div id="viewerContainer">

            <div id="viewer"></div>

            <button
                id="qm7xDownload3DBtn"
                class="qm7xDownloadBtn"
                title="Download the current 3D view as a PNG"
            >
                ⬇ PNG
            </button>

            <button
                id="qm7xDownloadXYZBtn"
                class="qm7xDownloadBtn"
                title="Download this molecule's XYZ coordinates"
            >
                ⬇ XYZ
            </button>

            {controls_html}

        </div>


        <script>

            // ==================================================
            // VIEWER
            // ==================================================

            const viewer =
                $3Dmol.createViewer(
                    "viewer",
                    {{

                        backgroundColor:
                            "{initial_background}",

                        antialias:
                            true

                    }}
                );


            // ==================================================
            // XYZ
            // ==================================================

            const xyz = `{xyz_js}`;

            console.log("3Dmol:", $3Dmol);
            console.log("Models:", viewer.getModelList());



            // ==================================================
            // LOAD MOLECULE
            // ==================================================

            viewer.addModel(
                xyz,
                "xyz"
            );


            // ==================================================
            // INITIAL STYLE
            // ==================================================

            viewer.setStyle(
                {{}},
                {style_js}
            );


            // ==================================================
            // AXES
            // ==================================================

            {axes_js}


            // ==================================================
            // APPLY PROJECTION
            // ==================================================

            function applyProjection(
                axis
            ) {{

                viewer.spin(false);

                viewer.setProjection(
                    axis === "free"
                        ? (
                            {str(orthographic).lower()}
                            ? "orthographic"
                            : "perspective"
                        )
                        : (
                            document.getElementById(
                                "orthographicCheck"
                            ) &&
                            document.getElementById(
                                "orthographicCheck"
                            ).checked
                                ? "orthographic"
                                : "perspective"
                        )
                );


                if (axis === "x") {{

                    viewer.setView([
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.70710678,
                        0,
                        0.70710678
                    ]);

                }}

                else if (axis === "y") {{

                    viewer.setView([
                        0,
                        0,
                        0,
                        0,
                        -0.70710678,
                        0,
                        0,
                        0.70710678
                    ]);

                }}

                else if (axis === "z") {{

                    viewer.setView([
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1
                    ]);

                }}

                else {{

                    viewer.setView([
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1
                    ]);

                }}



                viewer.zoomTo();

                viewer.render();

            }}


            // ==================================================
            // LABEL UPDATE
            // ==================================================
            //
            // labelsEnabled is the single source of truth for
            // whether atom labels are shown. It is seeded from the
            // Python `show_labels` argument (so it is correct even
            // when show_controls=False and #labelsCheck never
            // exists in the DOM), and kept in sync FROM the embedded
            // checkbox whenever that checkbox does exist -- so both
            // the no-controls call site (Molecular Analysis, in
            // tab_pairwise) and any future show_controls=True call
            // site stay correct.

            let labelsEnabled = {initial_labels_enabled_js};

            function updateLabels() {{

                viewer.removeAllLabels();

                const labelsCheck =
                    document.getElementById(
                        "labelsCheck"
                    );

                if (labelsCheck) {{
                    labelsEnabled = labelsCheck.checked;
                }}

                if (!labelsEnabled) {{
                    viewer.render();
                    return;
                }}

                const backgroundSelect =
                    document.getElementById(
                        "backgroundSelect"
                    );

                const backgroundValue =
                    backgroundSelect
                        ? backgroundSelect.value
                        : "{background}";

                const color =
                    backgroundValue === "white"
                        ? "green"
                        : "{label_color}";


                // Get the molecule model
                const model =
                    viewer.getModel();


                if (!model) {{
                    return;
                }}


                // Add labels to all atoms
                model.selectedAtoms({{}}).forEach(
                    function(atom) {{

                        viewer.addLabel(
                            atom.elem,
                            {{
                                position: {{
                                    x: atom.x,
                                    y: atom.y,
                                    z: atom.z
                                }},

                                font: "sans-serif",

                                fontSize: 17,

                                fontColor:
                                    color,

                                fontOpacity: 1,

                                backgroundOpacity: 1.9,

                                showBackground: false,

                                inFront: true,

                                alignment: "center"
                            }}
                        );

                    }}
                );


                viewer.render();

            }}

            // ==================================================
            // STYLE UPDATE
            // ==================================================

            function updateRepresentation() {{

                const representationSelect =
                    document.getElementById(
                        "representationSelect"
                    );


                if (!representationSelect) {{
                    return;
                }}


                const rep =
                    representationSelect.value;


                const atomScaleControl =
                    document.getElementById(
                        "atomScale"
                    );


                const bondRadiusControl =
                    document.getElementById(
                        "bondRadius"
                    );


                const atomScale =
                    atomScaleControl
                        ? parseFloat(
                            atomScaleControl.value
                        )
                        : {atom_scale};


                const bondRadius =
                    bondRadiusControl
                        ? parseFloat(
                            bondRadiusControl.value
                        )
                        : {bond_radius};


                let style = {{}};


                if (
                    rep ===
                    "Stick + Sphere"
                ) {{

                    style = {{

                        stick: {{
                            radius:
                                bondRadius
                        }},

                        sphere: {{
                            scale:
                                atomScale
                        }}

                    }};

                }}

                else if (
                    rep === "Stick"
                ) {{

                    style = {{

                        stick: {{
                            radius:
                                bondRadius
                        }}

                    }};

                }}

                else if (
                    rep === "Sphere"
                ) {{

                    style = {{

                        sphere: {{
                            scale:
                                atomScale
                        }}

                    }};

                }}

                else {{

                    style = {{
                        line: {{}}
                    }};

                }}


                viewer.setStyle(
                    {{}},
                    style
                );

                updateLabels();

                viewer.render();

            }}


            // ==================================================
            // BACKGROUND UPDATE
            // ==================================================

            function updateBackground() {{

                const backgroundSelect =
                    document.getElementById(
                        "backgroundSelect"
                    );


                if (!backgroundSelect) {{
                    return;
                }}


                const value =
                    backgroundSelect.value;


                if (value === "black") {{

                    viewer.setBackgroundColor(
                        "#000000"
                    );

                }}

                else if (value === "white") {{

                    viewer.setBackgroundColor(
                        "#FFFFFF"
                    );

                }}

                else {{

                    viewer.setBackgroundColor(
                        "rgba(0,0,0,0)"
                    );

                }}


                updateLabels();

                viewer.render();

            }}


            // ==================================================
            // ORTHOGRAPHIC
            // ==================================================

            function updateOrthographic() {{

                const checkbox =
                    document.getElementById(
                        "orthographicCheck"
                    );


                if (!checkbox) {{
                    return;
                }}


                viewer.setProjection(
                    checkbox.checked
                        ? "orthographic"
                        : "perspective"
                );


                viewer.render();

            }}


            // ==================================================
            // SPIN
            // ==================================================

            function updateSpin() {{

                const spinCheck =
                    document.getElementById(
                        "spinCheck"
                    );


                if (!spinCheck) {{
                    return;
                }}


                if (spinCheck.checked) {{

                    viewer.spin(
                        "y",
                        0.5
                    );

                }}

                else {{

                    viewer.spin(false);

                }}

            }}


            // ==================================================
            // BACKGROUND CONTROL
            // ==================================================

            const backgroundSelect =
                document.getElementById(
                    "backgroundSelect"
                );


            if (backgroundSelect) {{

                backgroundSelect.addEventListener(
                    "change",
                    updateBackground
                );

            }}


            // ==================================================
            // REPRESENTATION CONTROL
            // ==================================================

            const representationSelect =
                document.getElementById(
                    "representationSelect"
                );


            if (representationSelect) {{

                representationSelect.addEventListener(
                    "change",
                    updateRepresentation
                );

            }}


            // ==================================================
            // ATOM SIZE
            // ==================================================

            const atomScaleControl =
                document.getElementById(
                    "atomScale"
                );


            if (atomScaleControl) {{

                atomScaleControl.addEventListener(
                    "input",
                    updateRepresentation
                );

            }}


            // ==================================================
            // BOND RADIUS
            // ==================================================

            const bondRadiusControl =
                document.getElementById(
                    "bondRadius"
                );


            if (bondRadiusControl) {{

                bondRadiusControl.addEventListener(
                    "input",
                    updateRepresentation
                );

            }}


            // ==================================================
            // LABELS
            // ==================================================

            const labelsCheck =
                document.getElementById(
                    "labelsCheck"
                );


            if (labelsCheck) {{

                labelsCheck.addEventListener(
                    "change",
                    function() {{

                        updateLabels();

                        viewer.render();

                    }}
                );

            }}


            // ==================================================
            // PROJECTION
            // ==================================================

            const projectionSelect =
                document.getElementById(
                    "projectionSelect"
                );


            if (projectionSelect) {{

                projectionSelect.addEventListener(
                    "change",
                    function(event) {{

                        applyProjection(
                            event.target.value
                        );

                    }}
                );

            }}


            // ==================================================
            // ORTHOGRAPHIC CONTROL
            // ==================================================

            const orthographicCheck =
                document.getElementById(
                    "orthographicCheck"
                );


            if (orthographicCheck) {{

                orthographicCheck.addEventListener(
                    "change",
                    function() {{

                        const projectionSelect =
                            document.getElementById(
                                "projectionSelect"
                            );


                        if (projectionSelect) {{

                            applyProjection(
                                projectionSelect.value
                            );

                        }}

                        else {{

                            updateOrthographic();

                        }}

                    }}
                );

            }}


            // ==================================================
            // SPIN CONTROL
            // ==================================================

            const spinCheck =
                document.getElementById(
                    "spinCheck"
                );


            if (spinCheck) {{

                spinCheck.addEventListener(
                    "change",
                    updateSpin
                );

            }}


            // ==================================================
            // RESET
            // ==================================================

            const resetButton =
                document.getElementById(
                    "resetButton"
                );


            if (resetButton) {{

                resetButton.addEventListener(
                    "click",
                    function() {{

                        viewer.spin(false);


                        viewer.setProjection(
                            {str(orthographic).lower()}
                                ? "orthographic"
                                : "perspective"
                        );


                        const initialProjection =
                            "{projection}";


                        if (
                            initialProjection === "x"
                        ) {{

                            viewer.setView([
                                0,
                                0,
                                0,
                                0,
                                0,
                                0.70710678,
                                0,
                                0.70710678
                            ]);

                        }}

                        else if (
                            initialProjection === "y"
                        ) {{

                            viewer.setView([
                                0,
                                0,
                                0,
                                0,
                                -0.70710678,
                                0,
                                0,
                                0.70710678
                            ]);

                        }}

                        else {{

                            viewer.setView([
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                1
                            ]);

                        }}


                        viewer.zoomTo();

                        updateLabels();

                        viewer.render();

                    }}
                );

            }}


            // ==================================================
            // XYZ EXPORT
            // ==================================================

            const xyzButton =
                document.getElementById(
                    "xyzButton"
                );


            if (xyzButton) {{

                xyzButton.addEventListener(
                    "click",
                    function() {{

                        const blob =
                            new Blob(
                                [xyz],
                                {{

                                    type:
                                        "chemical/x-xyz"

                                }}
                            );


                        const url =
                            URL.createObjectURL(
                                blob
                            );


                        const link =
                            document.createElement(
                                "a"
                            );


                        link.href =
                            url;


                        link.download =
                            "molecule.xyz";


                        document.body.appendChild(
                            link
                        );


                        link.click();


                        document.body.removeChild(
                            link
                        );


                        URL.revokeObjectURL(
                            url
                        );

                    }}
                );

            }}


            // ==================================================
            // PNG EXPORT
            // ==================================================

            const pngButton =
                document.getElementById(
                    "pngButton"
                );


            if (pngButton) {{

                pngButton.addEventListener(
                    "click",
                    function() {{

                        try {{

                            const image =
                                viewer.pngURI();


                            const link =
                                document.createElement(
                                    "a"
                                );


                            link.href =
                                image;


                            link.download =
                                "molecule_3d.png";


                            document.body.appendChild(
                                link
                            );


                            link.click();


                            document.body.removeChild(
                                link
                            );

                        }}

                        catch (error) {{

                            console.error(
                                "PNG export failed:",
                                error
                            );

                        }}

                    }}
                );

            }}


            // ==================================================
            // INITIALIZE AXES
            // ==================================================

            if ({str(show_axes).lower()}) {{

                drawAxes();

            }}


            // ==================================================
            // INITIAL PROJECTION
            // ==================================================

            viewer.setProjection(
                {str(orthographic).lower()}
                    ? "orthographic"
                    : "perspective"
            );


            if ("{projection}" === "x") {{

                viewer.setView([
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.70710678,
                    0,
                    0.70710678
                ]);

            }}

            else if ("{projection}" === "y") {{

                viewer.setView([
                    0,
                    0,
                    0,
                    0,
                    -0.70710678,
                    0,
                    0,
                    0.70710678
                ]);

            }}

            else if ("{projection}" === "z") {{

                viewer.setView([
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1
                ]);

            }}

            else {{

                viewer.setView([
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1
                ]);

            }}


            // ==================================================
            // INITIAL CENTER
            // ==================================================
            //
            // (A leftover debug placeholder -- a hardcoded
            // viewer.addLabel("TEST", ...) at the origin, added
            // unconditionally on every load regardless of any
            // parameter -- used to sit here. Removed: it caused a
            // visible red "TEST" flash on load and was unrelated to
            // the show_labels/atom-label system entirely.)

            viewer.zoomTo();

            viewer.render();


            // ==================================================
            // INITIAL SPIN
            // ==================================================

            {spin_js}


            // ==================================================
            // INITIAL LABELS
            // ==================================================

            updateLabels();

            // ==================================================
            // ATOM CLICK BRIDGE (see definition above)
            // ==================================================

            {atom_selection_js}

            // ==================================================
            // INITIAL RENDER
            // ==================================================

            viewer.render();


            // ==================================================
            // DOWNLOAD 3D VIEW (always available, independent of
            // show_controls/enable_atom_selection)
            //
            // A one-way, client-side-only action -- no Python
            // round-trip, unlike the atom-click bridge above.
            // viewer.pngURI() returns a base64 PNG data URI of
            // whatever is currently rendered (current rotation,
            // zoom, representation, and background all included),
            // which a temporary <a download> element then saves.
            // ==================================================

            const qm7xDownloadBtn =
                document.getElementById(
                    "qm7xDownload3DBtn"
                );

            if (qm7xDownloadBtn) {{

                qm7xDownloadBtn.addEventListener(
                    "click",
                    function() {{

                        try {{

                            const dataUri = viewer.pngURI();

                            const link =
                                document.createElement("a");

                            link.href = dataUri;
                            link.download = "{download_filename_js}";

                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);

                        }} catch (err) {{
                            console.warn(
                                "qm7x 3D PNG export failed:",
                                err
                            );
                        }}
                    }}
                );
            }}


            // ==================================================
            // DOWNLOAD XYZ (same Blob + <a download> pattern
            // already used by display_comparative_3d_viewers()'s
            // own XYZ button -- reused verbatim rather than
            // reinvented. `xyz_js` is the exact XYZ text already
            // used to build this viewer's model above, so this is
            // guaranteed to match what's on screen.)
            // ==================================================

            const qm7xDownloadXYZBtn =
                document.getElementById(
                    "qm7xDownloadXYZBtn"
                );

            if (qm7xDownloadXYZBtn) {{

                qm7xDownloadXYZBtn.addEventListener(
                    "click",
                    function() {{

                        try {{

                            const blob = new Blob(
                                [`{xyz_js}`],
                                {{ type: "chemical/x-xyz" }}
                            );

                            const url =
                                URL.createObjectURL(blob);

                            const link =
                                document.createElement("a");

                            link.href = url;
                            link.download = "{download_filename_xyz_js}";

                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);

                            URL.revokeObjectURL(url);

                        }} catch (err) {{
                            console.warn(
                                "qm7x XYZ export failed:",
                                err
                            );
                        }}
                    }}
                );
            }}


            // ==================================================
            // RESIZE
            // ==================================================

            window.addEventListener(
                "resize",
                function() {{

                    viewer.resize();

                    viewer.render();

                }}
            );

        </script>

    </body>

    </html>
    """

    # ========================================================
    # STREAMLIT COMPONENT
    # ========================================================

    components.html(
        html,
        height=height,
        scrolling=False,
    )

# ============================================================
# COMPARATIVE 3D MOLECULAR VIEWERS
# ============================================================

def display_comparative_3d_viewers(comparison_molecules):               
    """
    Display multiple selected molecules in synchronized 3D viewers.

    Features
    --------
    - One shared control panel for all molecules
    - Stick + Sphere / Stick / Sphere / Line    
    - Atom labels
    - Atom size
    - Bond radius
    - Black / white / transparent background
    - Free 3D / X / Y / Z projection
    - Orthographic projection
    - Simultaneous auto-rotation
    - Reset all views
    - PNG export for individual molecules
    - XYZ export for individual molecules
    """

    if not comparison_molecules:
        st.info("No molecules available for 3D comparison.")
        return

    # ========================================================
    # PREPARE MOLECULE DATA
    # ========================================================

    molecules_js = []

    for index, item in enumerate(comparison_molecules):

        try:

            xyz = molecule_to_xyz(
                item["molecule"]
            )

        except Exception as e:

            st.warning(
                f"Could not convert molecule "
                f"{item['id']} to XYZ: {e}"
            )

            continue

        molecules_js.append(
            {
                "index": index,
                "id": str(
                    item["id"]
                ),
                "formula": str(
                    item.get(
                        "formula",
                        ""
                    )
                ),
                "xyz": xyz,
            }
        )


    if not molecules_js:

        st.error(
            "No molecules could be prepared "
            "for 3D visualization."
        )

        return


    # ========================================================
    # SERIALIZE MOLECULE DATA
    # ========================================================

    import json

    js_molecules = json.dumps(
        molecules_js,
        ensure_ascii=False,
    )


    # ========================================================
    # NUMBER OF COLUMNS
    # ========================================================

    n_molecules = len(molecules_js)

    if n_molecules <= 3:
        n_columns = n_molecules

    else:
        n_columns = 3

    # ========================================================
    # PYTHON-SIDE VIEWER HEIGHT
    # ========================================================

    viewer_height = 360

    card_height = 425

    n_rows = int(
        np.ceil(
            n_molecules / n_columns
        )
    )

    total_height = (
        125
        + (n_rows * card_height)
        + ((n_rows - 1) * 10)
        + 20
    )

    # ========================================================
    # HTML
    # ========================================================

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="UTF-8">

        <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>

        <style>

            html,
            body {{

                margin: 0;
                padding: 0;

                width: 100%;
                height: 100%;

                overflow: auto;

                background: transparent;

                font-family:
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    sans-serif;
            }}


            #mainContainer {{

                width: 100%;

                min-height: 100%;

                background: transparent;
            }}


            /* =================================================
               GLOBAL CONTROLS
               ================================================= */

            #controls {{

                position: relative;

                width: calc(100% - 16px);

                margin: 8px;

                box-sizing: border-box;

                padding: 10px 12px;

                border-radius: 9px;

                background:
                    rgba(15, 23, 42, 0.96);

                border:
                    1px solid
                    rgba(148, 163, 184, 0.25);

                color: #F8FAFC;

                backdrop-filter: blur(8px);

                font-size: 11px;
            }}


            .control-row {{

                display: flex;

                align-items: center;

                flex-wrap: wrap;

                gap: 10px;

                margin-bottom: 7px;
            }}


            .control-row:last-child {{

                margin-bottom: 0;
            }}


            .control-group {{

                display: flex;

                align-items: center;

                gap: 5px;

                white-space: nowrap;
            }}


            .control-title {{

                font-weight: 600;

                color: #CBD5E1;
            }}


            select {{

                background: #111827;

                color: #F8FAFC;

                border:
                    1px solid #334155;

                border-radius: 5px;

                padding: 4px 7px;

                font-size: 11px;

                outline: none;
            }}


            select:focus {{

                border-color: #19DBDE;
            }}


            input[type="range"] {{

                width: 75px;
            }}


            input[type="checkbox"] {{

                accent-color: #19DBDE;
            }}


            button {{

                background: #111827;

                color: #F8FAFC;

                border:
                    1px solid #334155;

                border-radius: 5px;

                padding: 5px 9px;

                font-size: 11px;

                cursor: pointer;

                transition:
                    background 0.15s,
                    border-color 0.15s;
            }}


            button:hover {{

                background: #1E293B;

                border-color: #19DBDE;
            }}


            .primary-button {{

                border-color:
                    rgba(25,219,222,0.55);
            }}


            /* =================================================
               MOLECULE GRID
               ================================================= */

            #moleculeGrid {{

                display: grid;

                grid-template-columns:
                    repeat({n_columns}, minmax(0, 1fr));

                gap: 10px;

                padding: 0 8px 8px 8px;

                box-sizing: border-box;
            }}


            .moleculeCard {{

                min-width: 0;

                border:
                    1px solid
                    rgba(148,163,184,0.18);

                border-radius: 9px;

                overflow: hidden;

                background:
                    rgba(15,23,42,0.30);
            }}


            .moleculeHeader {{

                padding: 8px 10px;

                background:
                    rgba(15,23,42,0.80);

                border-bottom:
                    1px solid
                    rgba(148,163,184,0.15);

                color: #F8FAFC;
            }}


            .moleculeFormula {{

                font-size: 15px;

                font-weight: 650;

                margin-bottom: 2px;
            }}


            .moleculeId {{

                font-size: 10px;

                color: #94A3B8;
            }}


            .viewerContainer {{

                width: 100%;

                height: {viewer_height}px;

                position: relative;

                background: #000000;
            }}


            .viewer {{

                width: 100%;

                height: 100%;

                position: absolute;

                left: 0;

                top: 0;
            }}


            .moleculeActions {{

                display: flex;

                gap: 6px;

                padding: 7px;

                background:
                    rgba(15,23,42,0.80);
            }}


            .moleculeActions button {{

                flex: 1;

                font-size: 10px;
            }}


            @media (max-width: 800px) {{

                #moleculeGrid {{

                    grid-template-columns:
                        repeat(2, minmax(0, 1fr));
                }}
            }}


            @media (max-width: 550px) {{

                #moleculeGrid {{

                    grid-template-columns:
                        1fr;
                }}
            }}

        </style>

    </head>


    <body>

        <div id="mainContainer">

            <!-- =============================================
                 SHARED CONTROLS
                 ============================================= -->

            <div id="controls">

                <div class="control-row">

                    <div class="control-group">

                        <span class="control-title">
                            Projection
                        </span>

                        <select id="projectionSelect">

                            <option value="free">
                                Free 3D
                            </option>

                            <option value="x">
                                X
                            </option>

                            <option value="y">
                                Y
                            </option>

                            <option value="z">
                                Z
                            </option>

                        </select>

                    </div>


                    <div class="control-group">

                        <span class="control-title">
                            Background
                        </span>

                        <select id="backgroundSelect">

                            <option value="black">
                                Black
                            </option>

                            <option value="white">
                                White
                            </option>

                            <option value="transparent">
                                Transparent
                            </option>

                        </select>

                    </div>


                    <div class="control-group">

                        <span class="control-title">
                            Representation
                        </span>

                        <select id="representationSelect">

                            <option value="Stick + Sphere">
                                Stick + Sphere
                            </option>

                            <option value="Stick">
                                Stick
                            </option>

                            <option value="Sphere">
                                Sphere
                            </option>

                            <option value="Line">
                                Line
                            </option>

                        </select>

                    </div>

                </div>


                <div class="control-row">

                    <div class="control-group">

                        <span class="control-title">
                            Atom size
                        </span>

                        <input
                            id="atomScale"
                            type="range"
                            min="0.15"
                            max="0.60"
                            step="0.01"
                            value="0.30"
                        >

                    </div>


                    <div class="control-group">

                        <span class="control-title">
                            Bond radius
                        </span>

                        <input
                            id="bondRadius"
                            type="range"
                            min="0.03"
                            max="0.30"
                            step="0.01"
                            value="0.12"
                        >

                    </div>


                    <label class="control-group">

                        <input
                            id="labelsCheck"
                            type="checkbox"
                            checked
                        >

                        Atom labels

                    </label>


                    <label class="control-group">

                        <input
                            id="orthographicCheck"
                            type="checkbox"
                        >

                        Orthographic

                    </label>


                    <label class="control-group">

                        <input
                            id="spinCheck"
                            type="checkbox"
                        >

                        Auto rotate

                    </label>


                    <button
                        id="resetButton"
                        class="primary-button"
                    >
                        ↻ Reset all
                    </button>

                </div>

            </div>


            <!-- =============================================
                 MOLECULE GRID
                 ============================================= -->

            <div id="moleculeGrid"></div>

        </div>


        <script>

            // =================================================
            // MOLECULE DATA
            // =================================================

            const molecules =
                {js_molecules};


            // =================================================
            // VIEWERS
            // =================================================

            const viewers = [];


            // =================================================
            // GLOBAL STATE
            // =================================================

            let currentBackground = "black";

            let currentProjection = "free";

            let currentRepresentation =
                "Stick + Sphere";

            let currentAtomScale = 0.30;

            let currentBondRadius = 0.12;

            let currentLabels = true;

            let currentOrthographic = false;

            let currentSpin = false;


            // =================================================
            // CREATE MOLECULE CARDS
            // =================================================

            const grid =
                document.getElementById(
                    "moleculeGrid"
                );


            molecules.forEach(
                function(molecule) {{

                    const card =
                        document.createElement(
                            "div"
                        );

                    card.className =
                        "moleculeCard";


                    // -----------------------------------------
                    // HEADER
                    // -----------------------------------------

                    const header =
                        document.createElement(
                            "div"
                        );

                    header.className =
                        "moleculeHeader";


                    const formula =
                        document.createElement(
                            "div"
                        );

                    formula.className =
                        "moleculeFormula";

                    formula.textContent =
                        molecule.formula;


                    const moleculeId =
                        document.createElement(
                            "div"
                        );

                    moleculeId.className =
                        "moleculeId";

                    moleculeId.textContent =
                        "Molecule ID: "
                        + molecule.id;


                    header.appendChild(
                        formula
                    );

                    header.appendChild(
                        moleculeId
                    );


                    // -----------------------------------------
                    // VIEWER
                    // -----------------------------------------

                    const viewerContainer =
                        document.createElement(
                            "div"
                        );

                    viewerContainer.className =
                        "viewerContainer";


                    const viewerElement =
                        document.createElement(
                            "div"
                        );

                    viewerElement.className =
                        "viewer";


                    viewerContainer.appendChild(
                        viewerElement
                    );


                    // -----------------------------------------
                    // ACTIONS
                    // -----------------------------------------

                    const actions =
                        document.createElement(
                            "div"
                        );

                    actions.className =
                        "moleculeActions";


                    const pngButton =
                        document.createElement(
                            "button"
                        );

                    pngButton.textContent =
                        "↓ PNG";


                    const xyzButton =
                        document.createElement(
                            "button"
                        );

                    xyzButton.textContent =
                        "↓ XYZ";


                    actions.appendChild(
                        pngButton
                    );

                    actions.appendChild(
                        xyzButton
                    );


                    // -----------------------------------------
                    // BUILD CARD
                    // -----------------------------------------

                    card.appendChild(
                        header
                    );

                    card.appendChild(
                        viewerContainer
                    );

                    card.appendChild(
                        actions
                    );

                    grid.appendChild(
                        card
                    );


                    // -----------------------------------------
                    // CREATE 3DMOL VIEWER
                    // -----------------------------------------

                    const viewer =
                        $3Dmol.createViewer(
                            viewerElement,
                            {{
                                backgroundColor:
                                    "#000000",

                                antialias:
                                    true
                            }}
                        );


                    // -----------------------------------------
                    // LOAD XYZ
                    // -----------------------------------------

                    viewer.addModel(
                        molecule.xyz,
                        "xyz"
                    );


                    viewers.push(
                        viewer
                    );


                    // -----------------------------------------
                    // PNG EXPORT
                    // -----------------------------------------

                    pngButton.addEventListener(
                        "click",
                        function() {{

                            try {{

                                const image =
                                    viewer.pngURI();


                                const link =
                                    document.createElement(
                                        "a"
                                    );


                                link.href =
                                    image;


                                link.download =
                                    "molecule_"
                                    + molecule.id
                                    + "_3d.png";


                                document.body.appendChild(
                                    link
                                );


                                link.click();


                                document.body.removeChild(
                                    link
                                );

                            }}

                            catch(error) {{

                                console.error(
                                    "PNG export failed:",
                                    error
                                );

                            }}

                        }}
                    );


                    // -----------------------------------------
                    // XYZ EXPORT
                    // -----------------------------------------

                    xyzButton.addEventListener(
                        "click",
                        function() {{

                            const blob =
                                new Blob(
                                    [molecule.xyz],
                                    {{
                                        type:
                                            "chemical/x-xyz"
                                    }}
                                );


                            const url =
                                URL.createObjectURL(
                                    blob
                                );


                            const link =
                                document.createElement(
                                    "a"
                                );


                            link.href =
                                url;


                            link.download =
                                "molecule_"
                                + molecule.id
                                + ".xyz";


                            document.body.appendChild(
                                link
                            );


                            link.click();


                            document.body.removeChild(
                                link
                            );


                            URL.revokeObjectURL(
                                url
                            );

                        }}
                    );

                }}
            );


            // =================================================
            // STYLE FUNCTION
            // =================================================

            function updateStyle() {{

                viewers.forEach(
                    function(viewer) {{

                        let style = {{}};


                        if (
                            currentRepresentation
                            ===
                            "Stick + Sphere"
                        ) {{

                            style = {{

                                stick: {{
                                    radius:
                                        currentBondRadius
                                }},

                                sphere: {{
                                    scale:
                                        currentAtomScale
                                }}

                            }};

                        }}


                        else if (
                            currentRepresentation
                            ===
                            "Stick"
                        ) {{

                            style = {{

                                stick: {{
                                    radius:
                                        currentBondRadius
                                }}

                            }};

                        }}


                        else if (
                            currentRepresentation
                            ===
                            "Sphere"
                        ) {{

                            style = {{

                                sphere: {{
                                    scale:
                                        currentAtomScale
                                }}

                            }};

                        }}


                        else {{

                            style = {{
                                line: {{}}
                            }};

                        }}


                        viewer.setStyle(
                            {{}},
                            style
                        );


                        viewer.removeAllLabels();


                        if (
                            currentLabels
                        ) {{

                            const labelColor =
                                currentBackground
                                ===
                                "black"
                                ? "white"
                                : "black";


                            // addPropertyLabels(propertyName,
                            // selector, style) takes THREE
                            // arguments (confirmed directly against
                            // the 3Dmol.js source: `selectedAtoms
                            // (t)` is called with the SECOND
                            // argument as the selector). This call
                            // used to pass only two -- "elem" and
                            // the style object -- so the style
                            // object was being read as the
                            // SELECTOR. Since no atom has
                            // properties literally named fontSize/
                            // fontColor/etc., that selector matched
                            // zero atoms, so no label was ever
                            // added regardless of currentLabels'
                            // true/false state. Fixed by passing an
                            // empty selector ({{}}, meaning "all
                            // atoms") as the second argument and
                            // the style as the third. Each viewer
                            // in `viewers` (see the forEach above)
                            // holds its OWN separately-loaded
                            // model, so "elem" is read per-atom
                            // from that viewer's own model --
                            // correct per-molecule element mapping
                            // falls out of this for free, not from
                            // any position-based lookup.
                            viewer.addPropertyLabels(
                                "elem",
                                {{}},
                                {{

                                    fontSize: 12,

                                    fontColor:
                                        labelColor,

                                    backgroundOpacity:
                                        0,

                                    showBackground:
                                        false,

                                    inFront:
                                        true

                                }}
                            );

                        }}


                        viewer.render();

                    }}
                );

            }}


            // =================================================
            // BACKGROUND
            // =================================================

            function updateBackground() {{

                let color;


                if (
                    currentBackground
                    ===
                    "white"
                ) {{

                    color = "#FFFFFF";

                }}

                else if (
                    currentBackground
                    ===
                    "transparent"
                ) {{

                    color =
                        "rgba(0,0,0,0)";

                }}

                else {{

                    color = "#000000";

                }}


                viewers.forEach(
                    function(viewer) {{

                        viewer.setBackgroundColor(
                            color
                        );

                        viewer.render();

                    }}
                );


                updateStyle();

            }}


            // =================================================
            // PROJECTION
            // =================================================

            function applyProjection(
                axis
            ) {{

                currentProjection =
                    axis;


                viewers.forEach(
                    function(viewer) {{

                        // -------------------------------------
                        // Stop spinning during projection
                        // -------------------------------------

                        viewer.spin(false);


                        // -------------------------------------
                        // Projection type
                        // -------------------------------------

                        viewer.setProjection(
                            currentOrthographic
                            ? "orthographic"
                            : "perspective"
                        );


                        // -------------------------------------
                        // Reset to molecule
                        // -------------------------------------

                        viewer.zoomTo();


                        /*
                         * 3Dmol's view is represented by:
                         *
                         * [x, y, z, zoom, qx, qy, qz, qw]
                         *
                         * We use quaternions to orient all
                         * viewers consistently.
                         */

                        if (
                            axis === "x"
                        ) {{

                            // Looking approximately
                            // along X

                            viewer.setView([
                                0,
                                0,
                                0,
                                0,
                                0,
                                0.70710678,
                                0,
                                0.70710678
                            ]);

                        }}


                        else if (
                            axis === "y"
                        ) {{

                            // Looking approximately
                            // along Y

                            viewer.setView([
                                0,
                                0,
                                0,
                                0,
                                -0.70710678,
                                0,
                                0,
                                0.70710678
                            ]);

                        }}


                        else if (
                            axis === "z"
                        ) {{

                            // Looking along Z

                            viewer.setView([
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                1
                            ]);

                        }}


                        viewer.zoomTo();

                        viewer.render();

                    }}
                );

            }}


            // =================================================
            // ORTHOGRAPHIC / PERSPECTIVE
            // =================================================

            function updateProjectionType() {{

                viewers.forEach(
                    function(viewer) {{

                        viewer.setProjection(
                            currentOrthographic
                            ? "orthographic"
                            : "perspective"
                        );

                        viewer.render();

                    }}
                );

            }}


            // =================================================
            // SPIN
            // =================================================

            function updateSpin() {{

                viewers.forEach(
                    function(viewer) {{

                        if (
                            currentSpin
                        ) {{

                            viewer.spin(
                                "y",
                                0.5
                            );

                        }}

                        else {{

                            viewer.spin(
                                false
                            );

                        }}

                    }}
                );

            }}


            // =================================================
            // PROJECTION SELECT
            // =================================================

            document
                .getElementById(
                    "projectionSelect"
                )
                .addEventListener(
                    "change",
                    function(event) {{

                        applyProjection(
                            event.target.value
                        );

                    }}
                );


            // =================================================
            // BACKGROUND SELECT
            // =================================================

            document
                .getElementById(
                    "backgroundSelect"
                )
                .addEventListener(
                    "change",
                    function(event) {{

                        currentBackground =
                            event.target.value;

                        updateBackground();

                    }}
                );


            // =================================================
            // REPRESENTATION SELECT
            // =================================================

            document
                .getElementById(
                    "representationSelect"
                )
                .addEventListener(
                    "change",
                    function(event) {{

                        currentRepresentation =
                            event.target.value;

                        updateStyle();

                    }}
                );


            // =================================================
            // ATOM SIZE
            // =================================================

            document
                .getElementById(
                    "atomScale"
                )
                .addEventListener(
                    "input",
                    function(event) {{

                        currentAtomScale =
                            parseFloat(
                                event.target.value
                            );

                        updateStyle();

                    }}
                );


            // =================================================
            // BOND RADIUS
            // =================================================

            document
                .getElementById(
                    "bondRadius"
                )
                .addEventListener(
                    "input",
                    function(event) {{

                        currentBondRadius =
                            parseFloat(
                                event.target.value
                            );

                        updateStyle();

                    }}
                );


            // =================================================
            // LABELS
            // =================================================

            document
                .getElementById(
                    "labelsCheck"
                )
                .addEventListener(
                    "change",
                    function(event) {{

                        currentLabels =
                            event.target.checked;

                        updateStyle();

                    }}
                );


            // =================================================
            // ORTHOGRAPHIC
            // =================================================

            document
                .getElementById(
                    "orthographicCheck"
                )
                .addEventListener(
                    "change",
                    function(event) {{

                        currentOrthographic =
                            event.target.checked;

                        updateProjectionType();

                    }}
                );


            // =================================================
            // SPIN
            // =================================================

            document
                .getElementById(
                    "spinCheck"
                )
                .addEventListener(
                    "change",
                    function(event) {{

                        currentSpin =
                            event.target.checked;

                        updateSpin();

                    }}
                );


            // =================================================
            // RESET ALL
            // =================================================

            document
                .getElementById(
                    "resetButton"
                )
                .addEventListener(
                    "click",
                    function() {{

                        currentBackground =
                            "black";

                        currentProjection =
                            "free";

                        currentRepresentation =
                            "Stick + Sphere";

                        currentAtomScale =
                            0.30;

                        currentBondRadius =
                            0.12;

                        currentLabels =
                            true;

                        currentOrthographic =
                            false;

                        currentSpin =
                            false;


                        document
                            .getElementById(
                                "backgroundSelect"
                            )
                            .value =
                            "black";


                        document
                            .getElementById(
                                "projectionSelect"
                            )
                            .value =
                            "free";


                        document
                            .getElementById(
                                "representationSelect"
                            )
                            .value =
                            "Stick + Sphere";


                        document
                            .getElementById(
                                "atomScale"
                            )
                            .value =
                            "0.30";


                        document
                            .getElementById(
                                "bondRadius"
                            )
                            .value =
                            "0.12";


                        document
                            .getElementById(
                                "labelsCheck"
                            )
                            .checked =
                            true;


                        document
                            .getElementById(
                                "orthographicCheck"
                            )
                            .checked =
                            false;


                        document
                            .getElementById(
                                "spinCheck"
                            )
                            .checked =
                            false;


                        viewers.forEach(
                            function(viewer) {{

                                viewer.spin(false);

                                viewer.setProjection(
                                    "perspective"
                                );

                                viewer.setBackgroundColor(
                                    "#000000"
                                );

                                viewer.setView([
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1
                                ]);

                                viewer.zoomTo();

                                viewer.render();

                            }}
                        );


                        updateStyle();

                    }}
                );


            // =================================================
            // INITIAL STYLE
            // =================================================

            updateStyle();


            // =================================================
            // INITIAL PROJECTION
            // =================================================

            viewers.forEach(
                function(viewer) {{

                    viewer.setProjection(
                        "perspective"
                    );

                    viewer.zoomTo();

                    viewer.render();

                }}
            );


            // =================================================
            // RESIZE
            // =================================================

            window.addEventListener(
                "resize",
                function() {{

                    viewers.forEach(
                        function(viewer) {{

                            viewer.resize();

                            viewer.render();

                        }}
                    );

                }}
            );

        </script>

    </body>

    </html>
    """

    # ========================================================
    # STREAMLIT
    # ========================================================

    components.html(
        html,
        height=total_height,
        scrolling=False,
    )

