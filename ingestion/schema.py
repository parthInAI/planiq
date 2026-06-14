"""
PlanIQ — Document Schema
Every chunk in the knowledge base carries this metadata.
No chunk enters the system without passing PlanningChunk validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import date, datetime
from enum import Enum


class DocumentType(str, Enum):
    PRIMARY_ACT        = "primary_act"         # Planning and Development Act 2000 / 2024
    SECONDARY_SI       = "secondary_si"         # Statutory Instruments (S.I. numbers)
    MINISTERIAL_GUIDE  = "ministerial_guide"    # DHLGH guidelines
    NATIONAL_POLICY    = "national_policy"      # NPF 2040, NDP
    COUNCIL_DEVPLAN    = "council_devplan"      # 31 local authority development plans
    ABP_DECISION       = "abp_decision"         # An Coimisiún Pleanála inspector reports
    EXEMPTION_SCHEDULE = "exemption_schedule"   # Schedule 2 PDR 2001 exempted classes


class Jurisdiction(str, Enum):
    NATIONAL = "national"
    # All 31 Irish local authorities
    CARLOW          = "carlow"
    CAVAN           = "cavan"
    CLARE           = "clare"
    CORK_CITY       = "cork_city"
    CORK_COUNTY     = "cork_county"
    DONEGAL         = "donegal"
    DUBLIN_CITY     = "dublin_city"
    DUN_LAOGHAIRE   = "dun_laoghaire_rathdown"
    FINGAL          = "fingal"
    GALWAY_CITY     = "galway_city"
    GALWAY_COUNTY   = "galway_county"
    KERRY           = "kerry"
    KILDARE         = "kildare"
    KILKENNY        = "kilkenny"
    LAOIS           = "laois"
    LEITRIM         = "leitrim"
    LIMERICK        = "limerick"
    LONGFORD        = "longford"
    LOUTH           = "louth"
    MAYO            = "mayo"
    MEATH           = "meath"
    MONAGHAN        = "monaghan"
    OFFALY          = "offaly"
    ROSCOMMON       = "roscommon"
    SLIGO           = "sligo"
    SOUTH_DUBLIN    = "south_dublin"
    TIPPERARY       = "tipperary"
    WATERFORD       = "waterford"
    WESTMEATH       = "westmeath"
    WEXFORD         = "wexford"
    WICKLOW         = "wicklow"


class ConfidenceLevel(str, Enum):
    HIGH    = "high"    # Verbatim statute text, verified against irishstatutebook.ie
    MEDIUM  = "medium"  # Official guidance documents, ministerial circulars
    LOW     = "low"     # Summarised / interpreted content — always surface disclaimer


class PlanningChunk(BaseModel):
    """
    The atomic unit of the PlanIQ knowledge base.
    Every retrieved chunk carries this full metadata contract.
    """
    # Identity
    chunk_id:        str = Field(..., description="UUID for this chunk")
    source_doc_id:   str = Field(..., description="Parent document identifier")
    chunk_index:     int = Field(..., description="Position within parent document")

    # Content
    text:            str = Field(..., min_length=20, description="The chunk text")
    summary:         str = Field("", description="One-line summary for reranker context")

    # Provenance — the hallucination shield
    document_type:   DocumentType
    jurisdiction:    Jurisdiction
    source_title:    str = Field(..., description="Full document title")
    source_url:      str = Field("", description="Canonical URL for verification")
    section_ref:     str = Field("", description="Exact section/schedule/class reference e.g. 'Schedule 2 Class 1 PDR 2001'")
    si_number:       str = Field("", description="S.I. number if a statutory instrument e.g. 'S.I. No. 600 of 2001'")
    act_year:        Optional[int] = Field(None, description="Year of the parent Act")

    # Temporal — staleness protection
    effective_date:  Optional[date] = Field(None, description="Date this provision came into force")
    last_verified:   date           = Field(default_factory=date.today, description="Date chunk was last verified against source")
    superseded_by:   Optional[str]  = Field(None, description="If not None, this chunk is STALE — reference to replacing document")
    expiry_date:     Optional[date]  = Field(None, description="If set, chunk is invalid after this date")

    # Quality signals — fed into confidence scoring
    confidence:      ConfidenceLevel = ConfidenceLevel.MEDIUM
    is_verbatim:     bool = Field(False, description="True if text is exact statutory wording")
    verified_url:    bool = Field(False, description="True if source URL has been confirmed live")

    # Operational
    ingested_at:     datetime = Field(default_factory=datetime.now)
    version:         str = Field("1.0", description="Schema version for migration tracking")

    @field_validator("superseded_by")
    @classmethod
    def flag_stale(cls, v):
        if v is not None and v.strip():
            # Any chunk with a superseded_by value is stale — retrieval pipeline blocks it
            return v.strip()
        return None

    @property
    def is_stale(self) -> bool:
        """Hard gate: chunk must not enter retrieval if stale or expired."""
        if self.superseded_by:
            return True
        if self.expiry_date and date.today() > self.expiry_date:
            return True
        return False

    @property
    def days_since_verified(self) -> int:
        return (date.today() - self.last_verified).days

    @property
    def needs_reverification(self) -> bool:
        """Flag chunks not verified in 30 days for the regulation watcher agent."""
        return self.days_since_verified > 30

    def to_chroma_metadata(self) -> dict:
        """Serialise to ChromaDB-compatible flat dict (no nested objects)."""
        return {
            "chunk_id":       self.chunk_id,
            "source_doc_id":  self.source_doc_id,
            "chunk_index":    self.chunk_index,
            "summary":        self.summary,
            "document_type":  self.document_type.value,
            "jurisdiction":   self.jurisdiction.value,
            "source_title":   self.source_title,
            "source_url":     self.source_url,
            "section_ref":    self.section_ref,
            "si_number":      self.si_number,
            "act_year":       self.act_year or 0,
            "effective_date": self.effective_date.isoformat() if self.effective_date else "",
            "last_verified":  self.last_verified.isoformat(),
            "superseded_by":  self.superseded_by or "",
            "expiry_date":    self.expiry_date.isoformat() if self.expiry_date else "",
            "confidence":     self.confidence.value,
            "is_verbatim":    self.is_verbatim,
            "verified_url":   self.verified_url,
            "is_stale":       self.is_stale,
            "needs_reverification": self.needs_reverification,
        }
