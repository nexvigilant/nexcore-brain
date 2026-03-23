//! T1 Primitive Grounding
//!
//! Universal T1 primitives that patterns and beliefs can be grounded to.
//! Enables cross-domain transfer and primitive-first filtering.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Universal T1 primitives that patterns can be grounded to.
///
/// Every pattern in the Brain maps to exactly one T1 primitive,
/// enabling cross-domain transfer and primitive-first filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum T1Primitive {
    /// Ordered operations (iterators, chains, pipelines)
    Sequence,
    /// Transformation A → B (From/Into, map, type conversions)
    Mapping,
    /// Self-reference (enum, Box, Arc, recursive fns)
    Recursion,
    /// Encapsulated context (struct+impl, typestate, invariants)
    State,
    /// Absence of value ((), `Option::None`, Default)
    Void,
    /// Delimiters, transition points, or scale limits.
    Boundary,
    /// Rate of occurrence; iterative counting.
    Frequency,
    /// Simple presence; instantiation of being.
    Existence,
    /// Continuity of state through temporal duration.
    Persistence,
    /// Producer (Cause) and consequence (Effect).
    Causality,
    /// Logical test; predicate matching.
    Comparison,
    /// Numerical magnitude; amount or ratio.
    Quantity,
    /// Positional context; address or coordinate.
    Location,
    /// One-way state transition; entropy floor.
    Irreversibility,
    /// Exclusive disjunction; exactly one of N variants (enum, match).
    Sum,
}

impl fmt::Display for T1Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Sequence => "sequence",
            Self::Mapping => "mapping",
            Self::Recursion => "recursion",
            Self::State => "state",
            Self::Void => "void",
            Self::Boundary => "boundary",
            Self::Frequency => "frequency",
            Self::Existence => "existence",
            Self::Persistence => "persistence",
            Self::Causality => "causality",
            Self::Comparison => "comparison",
            Self::Quantity => "quantity",
            Self::Location => "location",
            Self::Irreversibility => "irreversibility",
            Self::Sum => "sum",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_t1_primitive_display() {
        assert_eq!(T1Primitive::Sequence.to_string(), "sequence");
        assert_eq!(T1Primitive::Mapping.to_string(), "mapping");
        assert_eq!(T1Primitive::Recursion.to_string(), "recursion");
        assert_eq!(T1Primitive::State.to_string(), "state");
        assert_eq!(T1Primitive::Void.to_string(), "void");
        assert_eq!(T1Primitive::Boundary.to_string(), "boundary");
        assert_eq!(T1Primitive::Causality.to_string(), "causality");
    }

    #[test]
    fn test_t1_primitive_serde_roundtrip() {
        let original = T1Primitive::Mapping;
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: T1Primitive = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, deserialized);
    }
}
