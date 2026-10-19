// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

use scarabaeus_rust::Adder;

#[test]
fn add_positive_numbers() {
    let adder = Adder::__new__(2.0, 3.0);
    assert_eq!(adder.add(), 5.0);
}

#[test]
fn add_negative_numbers() {
    let adder = Adder::__new__(-2.0, -3.0);
    assert_eq!(adder.add(), -5.0);
}

#[test]
fn add_with_zero() {
    let adder = Adder::__new__(0.0, 7.0);
    assert_eq!(adder.add(), 7.0);
}
